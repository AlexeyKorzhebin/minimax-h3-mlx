"""Face detection and a smoothed bounding-box track, for the face-refine pipeline's crop window.

**Why detect every Nth frame instead of every frame.** YuNet is cheap (a few milliseconds on CPU
at this resolution) but not free, and a crop window does not need to react to every single frame
of jitter -- it needs to *follow* the face. `detect_track` samples every `every`th frame, and this
module's whole job is turning those sparse, occasionally-missing samples into a value for *every*
frame: `FaceTrack.box(frame_idx)` never returns "no data", only ever a box.

**Why interpolation, then smoothing, not one or the other.** A face-refine crop that jumped
straight to each new YuNet detection every 5 frames would visibly step; interpolating between
detections removes the step but leaves the *detections themselves* jittery, since YuNet's box is a
few pixels different every time even on a nearly-static face. `np.interp` supplies the
interpolation (and, as a side effect used deliberately -- see `_build_track` -- constant
extrapolation on both tails, exactly what the design calls for), and `scipy.signal.savgol_filter`
smooths the result afterward, on center and size separately: a Savitzky-Golay filter follows an
accelerating/decelerating motion (unlike a moving average, which lags behind it) while still
damping frame-to-frame detector noise.

**Why the largest face, not the first.** A frame can have more than one face in it (a poster on the
wall, a second person walking through) even after YuNet's own NMS, and across the whole clip the
faces seen are not necessarily always the same set. `detect_track` re-identifies faces *across*
sampled frames by nearest-center association (`_associate_tracks`) into separate candidate tracks
-- one per face that appears anywhere in the clip -- and then keeps only the candidate whose median
box area is largest across its own detections (`_select_subject_track`): the subject being tracked
is assumed to be whichever face is the most prominent *as a whole*, not whichever box happens to be
biggest in any single frame (a bystander who briefly steps close to the camera should not steal the
track from the subject one frame later).

**Deliberately no dependency on `mlx`.** A face-refine pass runs once per already-decoded clip,
CPU-only, and must not pay for loading a 33B-parameter transformer to do it -- same reasoning as
`h3_48gb.queue`'s own module docstring. See `test_facetrack_module_does_not_import_mlx`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

#: A hand-bumped marker for this module's own detection/track semantics -- bump it whenever a
#: change here (a different YuNet score threshold, a different Savitzky-Golay window, ...) would
#: make a track computed before the change not reproducible after it. `h3_48gb.cli.FACETRACK_VERSION`
#: is a separate, CLI-side copy of this same number (Task 4 does not otherwise touch this module),
#: recorded in every face-refine report's `checkpoint_identity_extra`; `test_cli_facerefine.py`
#: pins the two together so a bump on one side that forgets the other is a red test, not a silently
#: stale report field.
TRACK_ALGO_VERSION = 1

#: Where the YuNet ONNX weights must live. Not configurable: this module has exactly one detector,
#: and a fixed path means every caller (and every error message) agrees on where to look, the same
#: way `cli.DEFAULT_CHECKPOINT` is a fixed path rather than a parameter threaded through every call.
YUNET_MODEL_PATH = Path.home() / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"

#: Raw-content URL for the weights `_load_detector` requires -- opencv_zoo's own copy, ~230 KB.
#: Quoted verbatim in the "file is missing" error so a human can copy-paste the download command
#: instead of guessing where the model comes from (see `cli.py`'s `checkpoint_not_found`/
#: `checkpoint_without_adaln`, which this error message is modeled on).
YUNET_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

#: `cv2.FaceDetectorYN.create` requires *some* input size at construction time even though
#: `_detect_faces` overrides it per frame with `setInputSize` (frames here are not all the same
#: shape the detector was constructed for, in general) -- this value is never actually used to
#: detect anything.
_PLACEHOLDER_INPUT_SIZE = (320, 320)

#: YuNet's own demo defaults (opencv_zoo). `score_threshold` is deliberately looser than the demo's
#: 0.9: this is a crop-window tracker, not a security gate, and a lower threshold trades a few more
#: false positives (which `_associate_tracks` and the track math both already have to tolerate,
#: since a false positive is just another short-lived candidate track, discarded by
#: `_select_subject_track` unless it happens to outscore the real subject on median area)
#: for fewer missed detections on a face that is turned, lit badly, or partly out of frame -- the
#: exact conditions a face-refine crop most needs to keep tracking through.
_SCORE_THRESHOLD = 0.8
_NMS_THRESHOLD = 0.3
_TOP_K = 5000


def _load_detector():
    """A ready-to-use `cv2.FaceDetectorYN`, or a `RuntimeError` naming the exact fix.

    The failure is deliberately a plain `RuntimeError`, not a project-wide error code: this module
    has no `CliError`/`ERROR_CODES` contract to join (see the module docstring -- it is meant to be
    usable without importing anything from `h3_48gb.cli`), so the honest thing is a message a human
    can act on immediately, with the full path and the exact command to fetch the file -- the same
    shape as `cli.py`'s `checkpoint_not_found`.
    """
    if not YUNET_MODEL_PATH.exists():
        raise RuntimeError(
            f"YuNet face detector weights not found at {YUNET_MODEL_PATH}. Download it "
            f"(~230 KB, from opencv_zoo) with:\n"
            f"  mkdir -p {YUNET_MODEL_PATH.parent} && "
            f"curl -fL -o {YUNET_MODEL_PATH} {YUNET_DOWNLOAD_URL}"
        )
    return cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH), "", _PLACEHOLDER_INPUT_SIZE,
        score_threshold=_SCORE_THRESHOLD, nms_threshold=_NMS_THRESHOLD, top_k=_TOP_K,
    )


def _detect_faces(detector, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Every face `detector` finds in one frame, as `(x, y, w, h)` boxes (top-left corner and
    extent -- YuNet's own convention, and the one `FaceTrack.box` also returns).

    `frame` is RGB uint8, `(H, W, 3)` -- the shape every decoded frame in this fork has (see
    `LazyMiniMaxH3Pipeline._decode_video`'s postprocessing tail) -- and is converted to BGR here
    because YuNet, like the rest of OpenCV, expects it. `setInputSize` is called on every frame
    rather than once at construction: nothing here guarantees every frame handed to `detect_track`
    is the same size (a caller could feed cropped or resized frames), and `detect` silently
    misbehaves against a size it was not told about.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(bgr)
    if faces is None:
        return []
    return [(float(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in faces]


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _associate_tracks(
    detections: list[tuple[int, list[tuple[float, float, float, float]]]],
) -> list[list[tuple[int, tuple[float, float, float, float]]]]:
    """Group every sampled frame's detections into candidate per-face tracks, by greedy nearest-
    center matching between consecutive sampled frames -- the association `detect_track` was
    missing (see facetrack review finding C1): without it, picking the biggest box in *each frame
    independently* silently mixes together whichever faces happen to be in frame, with no notion
    that "the largest box in frame 5" and "the largest box in frame 10" might be two different
    people.

    `detections` is `(frame_idx, boxes)` per sampled frame, in the order `detect_track` sampled
    them (already increasing by construction, but not assumed here). Each returned track is a list
    of `(frame_idx, box)` pairs, exactly the shape `_build_track` wants for one identity.

    The matching is deliberately simple -- Euclidean distance between box centers, greedy nearest-
    first, one match per detection per frame -- rather than IoU: consecutive samples are `every`
    frames apart, and jitter aside, a real face's box does not reliably overlap itself that far
    apart if the face or camera is moving, whereas its *center* moves by a bounded, roughly
    continuous amount. Distinct faces in a normal frame are far apart relative to that jitter, so
    nearest-center is enough to keep them apart without needing a hard gating distance: a
    detection only ever prefers a closer track over a farther one, and greedy assignment (closest
    pairs claimed first) keeps one open track from stealing two detections in the same frame.

    An open track that gets no match in a given sampled frame is simply left waiting -- its last
    known box stays the matching target for future frames -- rather than closed off, since a face
    can drop out for a few samples (turned away, occluded) and reappear closer to where it was than
    to any other track.
    """
    tracks: list[list[tuple[int, tuple[float, float, float, float]]]] = []

    for frame_idx, boxes in detections:
        if not boxes:
            continue
        # Candidate (track, box) pairs, nearest first -- greedy assignment below claims the
        # closest pairs first so two detections in the same frame cannot both grab the same track.
        candidates = []
        for track_i, track in enumerate(tracks):
            last_center = _center(track[-1][1])
            for box_i, box in enumerate(boxes):
                dist = np.hypot(*(np.array(_center(box)) - np.array(last_center)))
                candidates.append((dist, track_i, box_i))
        candidates.sort(key=lambda item: item[0])

        matched_tracks: set[int] = set()
        matched_boxes: set[int] = set()
        for _dist, track_i, box_i in candidates:
            if track_i in matched_tracks or box_i in matched_boxes:
                continue
            matched_tracks.add(track_i)
            matched_boxes.add(box_i)
            tracks[track_i].append((frame_idx, boxes[box_i]))

        # Every unmatched box in this frame starts a new candidate track (a new face entering
        # frame, or -- on the very first sampled frame -- every face seen so far).
        for box_i, box in enumerate(boxes):
            if box_i not in matched_boxes:
                tracks.append([(frame_idx, box)])

    return tracks


def _select_subject_track(
    detections: list[tuple[int, list[tuple[float, float, float, float]]]],
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """The one candidate track, among all the faces `_associate_tracks` separated out, whose
    median box area is largest -- "the subject" of a face-refine crop, same "largest, but by
    median across the whole clip rather than a single frame" rule the module docstring's "why the
    largest face" note describes, now applied to a whole identity instead of one frame's boxes.

    Empty input (no detections anywhere) yields no tracks and therefore an empty result, which
    `_build_track` already treats as "no face to track" (`None`).
    """
    tracks = _associate_tracks(detections)
    if not tracks:
        return []

    def median_area(track: list[tuple[int, tuple[float, float, float, float]]]) -> float:
        areas = [box[2] * box[3] for _, box in track]
        return float(np.median(areas))

    return max(tracks, key=median_area)


def _smooth(series: np.ndarray, every: int) -> np.ndarray:
    """`series` (already interpolated to one value per frame) run through a Savitzky-Golay filter,
    or returned unsmoothed if there is not enough data to make that meaningful.

    The window spans two detection intervals (`2 * every + 1`, clamped to the series length and
    forced odd): wide enough to smooth out jitter *between* consecutive real detections -- a window
    of one interval would fit the filter almost exactly through every anchor and smooth nothing --
    without being so wide that a genuine, sustained camera or subject motion gets flattened into a
    lag, which is what a much larger window against `savgol_filter`'s own polynomial fit would do.

    `polyorder=2`: quadratic. A moving average (`polyorder=0`) lags behind acceleration; degree 1
    only follows constant velocity; degree 2 follows the acceleration/deceleration a face's motion
    (or a slow zoom) actually has, while still averaging away frame-to-frame detector jitter, which
    is high-frequency noise a low-degree polynomial cannot fit and therefore suppresses.

    Below `window_length=5` (four points more than the minimum `polyorder + 1 = 3` a quadratic fit
    needs, i.e. `window >= polyorder + 3`), smoothing would be barely distinguishable from the raw
    interpolation and not worth the discontinuity of switching behavior right at the edge of what
    `savgol_filter` accepts -- so short tracks (very few sampled frames) are returned interpolated
    but unsmoothed rather than raising or silently doing nothing.
    """
    window = min(len(series), 2 * every + 1)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return series
    return savgol_filter(series, window_length=window, polyorder=2)


@dataclass(frozen=True, eq=False)
class FaceTrack:
    """A face's bounding box, as a function of every frame index in `[0, n_frames)` -- built once,
    by `_build_track`, from however many real YuNet detections `detect_track` actually got.

    `_centers` (`(n_frames, 2)`, smoothed `(cx, cy)`) and `_sizes` (`(n_frames, 2)`, smoothed
    `(w, h)`) are stored separately, not as four smoothed scalar series or one smoothed `(x, y, w,
    h)`, because that is what was actually smoothed independently in `_build_track` -- center and
    size are physically different quantities (position vs. extent) with no reason to share a
    Savitzky-Golay fit, and reconstructing `(x, y)` from a smoothed *center* and a smoothed *size*
    keeps the box's center exactly where the smoothing put it regardless of how `w`/`h` moved,
    rather than smoothing `x`/`y` (the corner) and `w`/`h` independently and letting the corner
    drift relative to the center whenever size changes.

    `_near_detection` (`(n_frames,)` bool) backs `detected`; see that method for what "near" means.
    All three arrays are leading-underscore fields because they are this class's implementation,
    not its contract -- `box`, `detected`, `median_area` and `n_frames` are.
    """

    #: Total frames the source iterator yielded -- not just the ones YuNet ran on. `box`/`detected`
    #: accept any index in `[0, n_frames)`.
    n_frames: int
    #: Median area (`w * h`) of the *raw detections* `detect_track` kept (one, the largest, per
    #: sampled frame that found a face) -- deliberately not of the smoothed track, which would blur
    #: this "how big is this face, typically" summary with the constant-extrapolated tails.
    median_area: float
    _centers: np.ndarray
    _sizes: np.ndarray
    _near_detection: np.ndarray

    def _check_index(self, frame_idx: int) -> None:
        if not 0 <= frame_idx < self.n_frames:
            raise IndexError(f"frame_idx {frame_idx} out of range [0, {self.n_frames})")

    def box(self, frame_idx: int) -> tuple[float, float, float, float]:
        """`(x, y, w, h)` of the smoothed track at `frame_idx` -- the crop window's own coordinates,
        top-left corner and extent, always a value (see the module docstring): interpolated between
        real detections, constant-extrapolated before the first or after the last.
        """
        self._check_index(frame_idx)
        cx, cy = self._centers[frame_idx]
        w, h = self._sizes[frame_idx]
        return (float(cx - w / 2.0), float(cy - h / 2.0), float(w), float(h))

    def detected(self, frame_idx: int) -> bool:
        """Whether a real YuNet detection landed within one sampling interval (`every`, the
        argument `detect_track` was called with) of `frame_idx` -- not merely whether `frame_idx`
        falls between the first and last detection.

        This is what a caller doing fade-out wants: confidence should fall off gradually as the
        track moves away from its last real evidence, on *both* tails (interpolation's own middle
        stretch is always "near" by construction, since consecutive anchors are at most `every`
        frames apart) -- not flip from "trust it completely" to "trust it not at all" the instant
        the frame index crosses the outermost detection.
        """
        self._check_index(frame_idx)
        return bool(self._near_detection[frame_idx])


def _build_track(
    samples: list[tuple[int, tuple[float, float, float, float]]], n_frames: int, every: int
) -> FaceTrack | None:
    """The track-math core of `detect_track`, taking already-detected `(frame_idx, box)` samples
    instead of frames and a live detector -- split out so it can be tested without YuNet or even
    `cv2` (see `tests/test_facetrack.py`, which mocks the detector at this boundary).

    `None` if `samples` is empty: no detection anywhere in the source means there is no face to
    track, and a track that is a box everywhere by pure extrapolation from nothing would be a
    fabrication, not a fallback.
    """
    if not samples:
        return None

    samples = sorted(samples, key=lambda sample: sample[0])
    anchor_frames = [frame_idx for frame_idx, _ in samples]
    boxes = np.array([box for _, box in samples], dtype=np.float64)  # (k, 4): x, y, w, h
    areas = boxes[:, 2] * boxes[:, 3]
    median_area = float(np.median(areas))

    cx = boxes[:, 0] + boxes[:, 2] / 2.0
    cy = boxes[:, 1] + boxes[:, 3] / 2.0
    width = boxes[:, 2]
    height = boxes[:, 3]

    frame_idx = np.arange(n_frames, dtype=np.float64)
    anchor_idx = np.array(anchor_frames, dtype=np.float64)

    # `np.interp`'s default out-of-range behavior -- clamp to `fp[0]`/`fp[-1]` -- is exactly the
    # "constant extrapolation on the tails" the brief calls for, not a coincidence being relied on
    # silently: it is why `np.interp` was chosen over a hand-rolled interpolation here.
    centers = np.stack(
        [
            _smooth(np.interp(frame_idx, anchor_idx, cx), every),
            _smooth(np.interp(frame_idx, anchor_idx, cy), every),
        ],
        axis=1,
    )
    sizes = np.stack(
        [
            _smooth(np.interp(frame_idx, anchor_idx, width), every),
            _smooth(np.interp(frame_idx, anchor_idx, height), every),
        ],
        axis=1,
    )

    near_detection = np.zeros(n_frames, dtype=bool)
    for anchor in anchor_frames:
        lo = max(0, anchor - every)
        hi = min(n_frames - 1, anchor + every)
        near_detection[lo : hi + 1] = True

    return FaceTrack(
        n_frames=n_frames, median_area=median_area,
        _centers=centers, _sizes=sizes, _near_detection=near_detection,
    )


def detect_track(frames_iter, every: int = 5) -> FaceTrack | None:
    """Run YuNet on every `every`th frame of `frames_iter` and return the resulting `FaceTrack`,
    or `None` if not one sampled frame had a detectable face.

    `frames_iter` is fully consumed regardless of `every`: `FaceTrack.n_frames` (and therefore the
    domain `box`/`detected` accept) is the *total* frame count, not just the sampled ones, so the
    iterator is walked to its end even on frames YuNet never sees.

    The detector is loaded once, up front -- before the first frame is even pulled from
    `frames_iter` -- so a missing model file (`_load_detector`'s `RuntimeError`) fails immediately
    and predictably, the same "validate before doing any work" shape as `cli.RunSpec.__post_init__`,
    rather than however many frames into a possibly-expensive iterator.

    Every box YuNet finds on a sampled frame is kept (not just the largest -- see
    `_select_subject_track`/`_associate_tracks`): picking the largest *per frame* independently,
    with no association between frames, is exactly the bug facetrack review finding C1 describes --
    it silently mixes different faces together whenever their sizes are close, producing a track
    that sweeps across the whole clip instead of following one person.
    """
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")

    detector = _load_detector()
    detections: list[tuple[int, list[tuple[float, float, float, float]]]] = []
    n_frames = 0
    for frame_idx, frame in enumerate(frames_iter):
        n_frames = frame_idx + 1
        if frame_idx % every == 0:
            boxes = _detect_faces(detector, frame)
            if boxes:
                detections.append((frame_idx, boxes))

    samples = _select_subject_track(detections)
    return _build_track(samples, n_frames, every)
