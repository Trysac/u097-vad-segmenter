"""Pure-Python adaptive energy-based voice activity detection (VAD).

``StreamingSegmenter`` turns streamed audio samples (floats in [-1, 1]) into speech
``Segment`` objects; ``segment`` is the one-shot helper.

Energy
    The energy of a frame is ``20 * log10(rms)`` over the frame, floored at -100.0 dB.
    Full scale is a square wave at 1.0 (0 dB), so a full-scale sine is -3.01 dBFS. This is
    not the AES17 convention. Samples must be finite numbers; values outside [-1, 1] are
    clipped.

Algorithm
    1. Frames are consecutive non-overlapping blocks of ``sample_rate * frame_ms / 1000``
       samples. Leftover samples are buffered between ``feed`` calls; a partial final
       frame is dropped by ``flush``.
    2. A frame is loud when its energy is above ``noise_floor + start_margin_db`` and quiet
       when it is below ``noise_floor + stop_margin_db``. A frame between the margins is
       neither and breaks both the loud run and the quiet run.
    3. Per frame, in order: the calibration window; loud or quiet classification; then,
       while in speech, the ``max_segment_ms`` cut and the silence run, otherwise the loud
       run and the speech start; finally the noise floor update.
    4. The first ``calibration_ms`` of a stream never start speech and set the floor to
       their mean energy. Afterwards the floor is an exponential moving average updated
       only on frames outside speech that are not loud, clamped to [-100, -10] dB.
       Speech inside the calibration window is a documented limitation: it raises the floor.
    5. Speech starts after ``min_speech_ms`` of consecutive loud frames and is back-dated
       to the first of them. Speech ends after ``min_silence_ms`` of consecutive quiet
       frames; the segment ends at the end of the last loud frame (hysteresis).
    6. A segment includes ``pre_roll_ms`` of audio before its start, clipped at the stream
       start. A segment whose ``end_ms - start_ms`` (pre-roll included) reaches
       ``max_segment_ms`` is cut there with ``forced_split`` True and continues as a new
       segment without pre-roll. Memory stays within ``max_segment_ms`` plus pre-roll.

Timestamps are derived from sample counts. All errors are ``VadError`` with a stable code.
"""

from __future__ import annotations

import logging
import math
import struct
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Literal

__all__ = [
    "Segment",
    "StreamingSegmenter",
    "VadConfig",
    "VadError",
    "__version__",
    "frame_energy_db",
    "pcm16_to_float",
    "segment",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MAX_FRAME_SAMPLES = 1_000_000
MAX_FEED_SAMPLES = 10_000_000
MAX_SEGMENT_INPUT_SAMPLES = 100_000_000
MAX_PCM_BYTES = 20_000_000

_FEED_CHUNK = 65_536
_FLOOR_MIN_DB = -100.0
_FLOOR_MAX_DB = -10.0
_SILENCE_DB = -100.0

ErrorCode = Literal["invalid_config", "invalid_audio", "closed"]


class VadError(Exception):
    """Base error of the library.

    Attributes:
        code: stable machine-readable error code (``invalid_config``, ``invalid_audio``
            or ``closed``).
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message[:200])
        self.code: ErrorCode = code


def _check_int(name: str, value: object, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise VadError("invalid_config", f"{name} must be an integer in [{low}, {high}]")


def _check_choice(name: str, value: object, choices: tuple[int, ...]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in choices:
        raise VadError("invalid_config", f"{name} must be one of {list(choices)}")


def _check_real(
    name: str, value: object, low: float, high: float, *, exclusive_low: bool = False
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VadError("invalid_config", f"{name} must be a number")
    if not math.isfinite(value):
        raise VadError("invalid_config", f"{name} must be finite")
    below = value <= low if exclusive_low else value < low
    if below or value > high:
        raise VadError("invalid_config", f"{name} is out of range ({low}, {high})")


@dataclass(frozen=True)
class VadConfig:
    """Validated configuration of the detector.

    Durations are rounded up to whole frames. Raises ``VadError`` with code
    ``invalid_config`` when a value is invalid.
    """

    sample_rate: Literal[8000, 16000, 32000, 48000] = 16000
    frame_ms: Literal[10, 20, 30] = 20
    start_margin_db: float = 9.0
    stop_margin_db: float = 6.0
    min_speech_ms: int = 200
    min_silence_ms: int = 400
    pre_roll_ms: int = 200
    max_segment_ms: int = 15000
    noise_adapt_rate: float = 0.05
    initial_noise_db: float = -60.0
    calibration_ms: int = 300

    def __post_init__(self) -> None:
        _check_choice("sample_rate", self.sample_rate, (8000, 16000, 32000, 48000))
        _check_choice("frame_ms", self.frame_ms, (10, 20, 30))
        _check_real("start_margin_db", self.start_margin_db, 0.0, 100.0, exclusive_low=True)
        _check_real("stop_margin_db", self.stop_margin_db, 0.0, 100.0, exclusive_low=True)
        if self.stop_margin_db > self.start_margin_db:
            raise VadError("invalid_config", "stop_margin_db must not exceed start_margin_db")
        _check_int("min_speech_ms", self.min_speech_ms, 1, 120_000)
        _check_int("min_silence_ms", self.min_silence_ms, 1, 120_000)
        _check_int("pre_roll_ms", self.pre_roll_ms, 0, 1000)
        _check_int("max_segment_ms", self.max_segment_ms, 1000, 120_000)
        _check_real("noise_adapt_rate", self.noise_adapt_rate, 0.0, 1.0, exclusive_low=True)
        _check_real("initial_noise_db", self.initial_noise_db, -100.0, -10.0)
        _check_int("calibration_ms", self.calibration_ms, 0, 10_000)
        if self.min_speech_ms + self.pre_roll_ms > self.max_segment_ms:
            raise VadError("invalid_config", "min_speech_ms + pre_roll_ms exceeds max_segment_ms")
        if self.min_silence_ms > self.max_segment_ms:
            raise VadError("invalid_config", "min_silence_ms exceeds max_segment_ms")

    def _frames(self, milliseconds: int) -> int:
        return -(-milliseconds // self.frame_ms)

    @property
    def frame_samples(self) -> int:
        """Number of samples in one frame."""
        return self.sample_rate * self.frame_ms // 1000

    @property
    def min_speech_frames(self) -> int:
        """``min_speech_ms`` rounded up to frames."""
        return self._frames(self.min_speech_ms)

    @property
    def min_silence_frames(self) -> int:
        """``min_silence_ms`` rounded up to frames."""
        return self._frames(self.min_silence_ms)

    @property
    def pre_roll_frames(self) -> int:
        """``pre_roll_ms`` rounded up to frames."""
        return self._frames(self.pre_roll_ms)

    @property
    def calibration_frames(self) -> int:
        """``calibration_ms`` rounded up to frames."""
        return self._frames(self.calibration_ms)


@dataclass(frozen=True)
class Segment:
    """A speech segment.

    ``peak_db`` and ``mean_db`` are the maximum and the mean of the per-frame dBFS values
    of every frame in ``samples`` (pre-roll included).
    """

    start_ms: int
    end_ms: int
    samples: tuple[float, ...]
    forced_split: bool
    peak_db: float
    mean_db: float


def _too_many(limit: int) -> VadError:
    return VadError("invalid_audio", f"too many samples (limit {limit})")


def _sized(samples: Sequence[float]) -> int:
    if isinstance(samples, str | bytes | bytearray | memoryview):
        raise VadError("invalid_audio", "samples must be a sequence of numbers")
    try:
        return len(samples)
    except TypeError:
        raise VadError("invalid_audio", "samples must be a sized sequence") from None


def _clean(samples: Sequence[float], limit: int) -> list[float]:
    if _sized(samples) > limit:
        raise _too_many(limit)
    out: list[float] = []
    for value in samples:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise VadError("invalid_audio", "samples must be numbers")
        try:
            number = float(value)
        except OverflowError:
            raise VadError("invalid_audio", "sample is too large") from None
        if not math.isfinite(number):
            raise VadError("invalid_audio", "samples must be finite")
        out.append(max(-1.0, min(1.0, number)))
    return out


def _energy(frame: Sequence[float]) -> float:
    total = sum(x * x for x in frame)
    rms = math.sqrt(total / len(frame))
    if rms <= 0.0:
        return _SILENCE_DB
    return max(_SILENCE_DB, 20.0 * math.log10(rms))


def _clamp_floor(value: float) -> float:
    return max(_FLOOR_MIN_DB, min(_FLOOR_MAX_DB, value))


def frame_energy_db(frame: Sequence[float]) -> float:
    """Return the energy of ``frame`` in dBFS: ``20 * log10(rms)``, floored at -100.0.

    Full scale is a square wave at 1.0, so a full-scale sine is -3.01 dBFS. Values outside
    [-1, 1] are clipped.

    Raises:
        VadError: ``invalid_audio`` for an empty, oversized or non-finite frame.
    """
    values = _clean(frame, MAX_FRAME_SAMPLES)
    if not values:
        raise VadError("invalid_audio", "frame must not be empty")
    return _energy(values)


def pcm16_to_float(data: bytes) -> list[float]:
    """Convert little-endian signed 16-bit PCM to floats by dividing by 32768.

    Raises:
        VadError: ``invalid_audio`` for a wrong type, an odd length or too many bytes.
    """
    if not isinstance(data, bytes | bytearray):
        raise VadError("invalid_audio", "data must be bytes")
    if len(data) > MAX_PCM_BYTES:
        raise VadError("invalid_audio", f"too many bytes (limit {MAX_PCM_BYTES})")
    if len(data) % 2:
        raise VadError("invalid_audio", "PCM16 data must have an even number of bytes")
    values = struct.unpack(f"<{len(data) // 2}h", data)
    return [value / 32768.0 for value in values]


class StreamingSegmenter:
    """Incremental speech segmenter; see the module docstring for the algorithm."""

    def __init__(self, config: VadConfig | None = None) -> None:
        if config is None:
            config = VadConfig()
        elif not isinstance(config, VadConfig):
            raise VadError("invalid_config", "config must be a VadConfig")
        self._config = config
        self._frame_samples = config.frame_samples
        self._min_speech = config.min_speech_frames
        self._min_silence = config.min_silence_frames
        self._calibration_frames = config.calibration_frames
        self._floor = config.initial_noise_db
        self._cal_sum = 0.0
        self._pending: list[float] = []
        self._frame_idx = 0
        size = config.pre_roll_frames + config.min_speech_frames
        self._ring: deque[tuple[tuple[float, ...], float]] = deque(maxlen=size)
        self._in_speech = False
        self._loud_run = 0
        self._quiet_run = 0
        self._seg_start = 0
        self._seg_frames: list[tuple[float, ...]] = []
        self._seg_energies: list[float] = []
        self._last_loud = 0
        self._out: list[Segment] = []
        self._closed = False

    @property
    def noise_floor_db(self) -> float:
        """Current noise floor estimate in dBFS (read-only)."""
        return self._floor

    def feed(self, samples: Sequence[float]) -> list[Segment]:
        """Consume samples and return the segments completed by them.

        Raises:
            VadError: ``closed`` after ``flush``; ``invalid_audio`` for bad samples. A
                rejected call leaves the stream unchanged.
        """
        if self._closed:
            raise VadError("closed", "stream is closed")
        values = _clean(samples, MAX_FEED_SAMPLES)
        self._pending.extend(values)
        size = self._frame_samples
        count = len(self._pending) // size
        for index in range(count):
            start = index * size
            self._process_frame(tuple(self._pending[start : start + size]))
        del self._pending[: count * size]
        out, self._out = self._out, []
        return out

    def flush(self) -> list[Segment]:
        """End the stream, close an open segment and return it. Terminal.

        A partial final frame is dropped.

        Raises:
            VadError: ``closed`` when the stream was already flushed.
        """
        if self._closed:
            raise VadError("closed", "stream is closed")
        self._closed = True
        if self._in_speech:
            self._end_speech()
        self._pending.clear()
        out, self._out = self._out, []
        return out

    def _process_frame(self, frame: tuple[float, ...]) -> None:
        cfg = self._config
        energy = _energy(frame)
        index = self._frame_idx
        self._frame_idx += 1
        if index < self._calibration_frames:
            self._cal_sum += energy
            self._floor = _clamp_floor(self._cal_sum / (index + 1))
            self._ring.append((frame, energy))
            return
        loud = energy > self._floor + cfg.start_margin_db
        quiet = energy < self._floor + cfg.stop_margin_db
        if self._in_speech:
            self._step_speech(frame, energy, index, loud, quiet)
        else:
            self._ring.append((frame, energy))
            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self._min_speech:
                self._start_speech(index)
        if not self._in_speech and not loud:
            rate = cfg.noise_adapt_rate
            self._floor = _clamp_floor(self._floor + rate * (energy - self._floor))

    def _step_speech(
        self, frame: tuple[float, ...], energy: float, index: int, loud: bool, quiet: bool
    ) -> None:
        self._seg_frames.append(frame)
        self._seg_energies.append(energy)
        if loud:
            self._quiet_run = 0
            self._last_loud = len(self._seg_frames)
        elif quiet:
            self._quiet_run += 1
        else:
            self._quiet_run = 0
        elapsed = (index + 1 - self._seg_start) * self._config.frame_ms
        if elapsed >= self._config.max_segment_ms:
            self._cut(index + 1)
        if self._quiet_run >= self._min_silence:
            self._end_speech()

    def _start_speech(self, index: int) -> None:
        entries = list(self._ring)
        self._ring.clear()
        self._seg_frames = [entry[0] for entry in entries]
        self._seg_energies = [entry[1] for entry in entries]
        self._seg_start = index + 1 - len(entries)
        self._last_loud = len(entries)
        self._in_speech = True
        self._loud_run = 0
        self._quiet_run = 0

    def _cut(self, end_frame: int) -> None:
        segment_ = self._make_segment(self._seg_start, self._seg_frames, self._seg_energies, True)
        self._out.append(segment_)
        self._seg_start = end_frame
        self._seg_frames = []
        self._seg_energies = []
        self._last_loud = 0

    def _end_speech(self) -> None:
        count = self._last_loud
        if count > 0:
            frames = self._seg_frames[:count]
            energies = self._seg_energies[:count]
            self._out.append(self._make_segment(self._seg_start, frames, energies, False))
        tail = list(zip(self._seg_frames[count:], self._seg_energies[count:], strict=True))
        self._ring.clear()
        self._ring.extend(tail)
        self._seg_frames = []
        self._seg_energies = []
        self._last_loud = 0
        self._in_speech = False
        self._loud_run = 0
        self._quiet_run = 0

    def _make_segment(
        self,
        start_frame: int,
        frames: list[tuple[float, ...]],
        energies: list[float],
        forced: bool,
    ) -> Segment:
        frame_ms = self._config.frame_ms
        result = Segment(
            start_ms=start_frame * frame_ms,
            end_ms=(start_frame + len(frames)) * frame_ms,
            samples=tuple(chain.from_iterable(frames)),
            forced_split=forced,
            peak_db=max(energies),
            mean_db=sum(energies) / len(energies),
        )
        logger.debug("segment %d-%d ms forced=%s", result.start_ms, result.end_ms, forced)
        return result


def segment(samples: Sequence[float], config: VadConfig | None = None) -> list[Segment]:
    """Segment a complete recording; equivalent to feeding everything and flushing.

    Raises:
        VadError: ``invalid_config`` or ``invalid_audio``.
    """
    streamer = StreamingSegmenter(config)
    total = _sized(samples)
    if total > MAX_SEGMENT_INPUT_SAMPLES:
        raise _too_many(MAX_SEGMENT_INPUT_SAMPLES)
    out: list[Segment] = []
    for start in range(0, total, _FEED_CHUNK):
        out.extend(streamer.feed(samples[start : start + _FEED_CHUNK]))
    out.extend(streamer.flush())
    return out
