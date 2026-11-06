"""Tests for the vad_segmenter public API."""

from __future__ import annotations

import dataclasses
import logging
import math
import random

import pytest

import vad_segmenter
from vad_segmenter import (
    Segment,
    StreamingSegmenter,
    VadConfig,
    VadError,
    frame_energy_db,
    pcm16_to_float,
    segment,
)

RATE = 16000


def zeros(ms: int, rate: int = RATE) -> list[float]:
    return [0.0] * (rate * ms // 1000)


def sine(ms: int, amp: float = 0.5, freq: float = 440.0, rate: int = RATE) -> list[float]:
    count = rate * ms // 1000
    return [amp * math.sin(2 * math.pi * freq * n / rate) for n in range(count)]


def const(ms: int, level: float) -> list[float]:
    return [level] * (RATE * ms // 1000)


def spans(result: list[Segment]) -> list[tuple[int, int, bool]]:
    return [(s.start_ms, s.end_ms, s.forced_split) for s in result]


def chunked(samples: list[float], size: int, config: VadConfig | None = None) -> list[Segment]:
    streamer = StreamingSegmenter(config)
    out: list[Segment] = []
    for start in range(0, len(samples), size):
        out.extend(streamer.feed(samples[start : start + size]))
    out.extend(streamer.flush())
    return out


SIGNAL = zeros(500) + sine(600) + zeros(300) + sine(600) + zeros(500)


def test_single_segment_with_pre_roll() -> None:
    result = segment(zeros(1000) + sine(1000) + zeros(1000))
    assert len(result) == 1
    seg = result[0]
    assert seg.start_ms == 800
    assert seg.end_ms == 2000
    assert seg.forced_split is False
    assert len(seg.samples) == 19200
    assert seg.peak_db == pytest.approx(-9.03, abs=0.3)
    assert -100.0 < seg.mean_db < seg.peak_db


def test_short_burst_and_short_gap() -> None:
    assert segment(zeros(1000) + sine(100) + zeros(1000)) == []
    result = segment(zeros(1000) + sine(1000) + zeros(300) + sine(1000) + zeros(1000))
    assert spans(result) == [(800, 3300, False)]
    assert spans(segment(zeros(1000) + sine(200) + zeros(1000))) == [(800, 1200, False)]


@pytest.mark.parametrize("size", [1, 333])
@pytest.mark.parametrize("config", [None, VadConfig(max_segment_ms=1000)])
def test_chunking_does_not_change_segments(size: int, config: VadConfig | None) -> None:
    """Feeding in 1-sample chunks, 333-sample chunks or one call yields equal lists."""
    whole = chunked(SIGNAL, len(SIGNAL), config)
    assert whole
    assert chunked(SIGNAL, size, config) == whole
    if config is not None:
        assert any(s.forced_split for s in whole)


def test_forced_split_continuation_has_no_pre_roll() -> None:
    config = VadConfig(max_segment_ms=15000)
    result = segment(zeros(1000) + sine(20000) + zeros(1000), config)
    assert spans(result) == [(800, 15800, True), (15800, 21000, False)]
    assert len(result[0].samples) == 240000
    assert len(result[1].samples) == 83200


def test_noise_only_trains_the_floor() -> None:
    rng = random.Random(42)  # noqa: S311
    noise = [rng.uniform(-0.05477, 0.05477) for _ in range(5 * RATE)]
    streamer = StreamingSegmenter()
    assert streamer.feed(noise) == []
    assert streamer.flush() == []
    assert streamer.noise_floor_db == pytest.approx(-30.0, abs=0.5)


def test_frame_energy_db_reference_values() -> None:
    frame = sine(20, amp=1.0, freq=1000.0)
    assert len(frame) == 320
    assert frame_energy_db(frame) == pytest.approx(-3.0103, abs=0.0001)
    assert frame_energy_db([0.0] * 320) == -100.0


def test_frame_energy_db_edges() -> None:
    assert frame_energy_db([2.0] * 10) == 0.0
    assert frame_energy_db([1, -1]) == 0.0
    assert frame_energy_db([1e-200] * 4) == -100.0
    assert frame_energy_db([1e-8] * 4) == -100.0
    assert frame_energy_db([0.5] * 8) == pytest.approx(-6.0206, abs=0.001)
    with pytest.raises(VadError) as info:
        frame_energy_db([])
    assert info.value.code == "invalid_audio"


def test_frame_energy_db_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad_segmenter, "MAX_FRAME_SAMPLES", 4)
    with pytest.raises(VadError) as info:
        frame_energy_db([0.1] * 5)
    assert info.value.code == "invalid_audio"


def test_flush_closes_open_segment_and_is_terminal() -> None:
    """flush() closes an open segment; feed or flush afterwards raises code 'closed'."""
    streamer = StreamingSegmenter()
    assert streamer.feed(zeros(1000) + sine(1000)) == []
    closed = streamer.flush()
    assert spans(closed) == [(800, 2000, False)]
    with pytest.raises(VadError) as info:
        streamer.feed([0.0])
    assert info.value.code == "closed"
    with pytest.raises(VadError) as info:
        streamer.flush()
    assert info.value.code == "closed"


def test_empty_input_and_nan() -> None:
    assert segment([]) == []
    streamer = StreamingSegmenter()
    assert streamer.feed([]) == []
    with pytest.raises(VadError) as info:
        streamer.feed([float("nan")])
    assert info.value.code == "invalid_audio"
    # feed([float('nan')]) leaves the stream usable and unchanged
    assert streamer.feed(zeros(1000)) == []


@pytest.mark.parametrize(
    "bad",
    [
        ["a"],
        [True],
        [None],
        [float("inf")],
        [float("-inf")],
        [10**400],
        [0.0, 0.0, float("nan")],
    ],
)
def test_invalid_samples(bad: list[object]) -> None:
    with pytest.raises(VadError) as info:
        StreamingSegmenter().feed(bad)  # type: ignore[arg-type]
    assert info.value.code == "invalid_audio"


@pytest.mark.parametrize("bad", ["abc", b"ab", bytearray(b"ab"), 5, None])
def test_invalid_sample_containers(bad: object) -> None:
    with pytest.raises(VadError) as info:
        StreamingSegmenter().feed(bad)  # type: ignore[arg-type]
    assert info.value.code == "invalid_audio"
    with pytest.raises(VadError) as info:
        segment(bad)  # type: ignore[arg-type]
    assert info.value.code == "invalid_audio"


def test_integer_samples_and_clipping() -> None:
    assert segment([0] * 1000) == []
    result = segment(zeros(1000) + sine(1000, amp=5.0) + zeros(1000))
    assert len(result) == 1
    assert max(result[0].samples) == 1.0
    assert min(result[0].samples) >= -1.0


def test_input_size_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad_segmenter, "MAX_FEED_SAMPLES", 5)
    with pytest.raises(VadError) as info:
        StreamingSegmenter().feed([0.0] * 6)
    assert info.value.code == "invalid_audio"
    monkeypatch.setattr(vad_segmenter, "MAX_SEGMENT_INPUT_SAMPLES", 5)
    with pytest.raises(VadError) as info:
        segment([0.0] * 6)
    assert info.value.code == "invalid_audio"


def test_invalid_config_acceptance_literals() -> None:
    with pytest.raises(VadError) as info:
        VadConfig(frame_ms=25)  # type: ignore[arg-type]
    assert info.value.code == "invalid_config"
    with pytest.raises(VadError) as info:
        VadConfig(stop_margin_db=-1.0)
    assert info.value.code == "invalid_config"
    with pytest.raises(VadError) as info:
        VadConfig(pre_roll_ms=1001)
    assert info.value.code == "invalid_config"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 44100},
        {"sample_rate": True},
        {"frame_ms": True},
        {"frame_ms": 25},
        {"start_margin_db": 0.0},
        {"start_margin_db": float("nan")},
        {"start_margin_db": "9"},
        {"start_margin_db": 101.0},
        {"stop_margin_db": 0.0},
        {"stop_margin_db": 10.0},
        {"min_speech_ms": 0},
        {"min_speech_ms": 20.5},
        {"min_silence_ms": 0},
        {"pre_roll_ms": -1},
        {"max_segment_ms": 999},
        {"max_segment_ms": 120001},
        {"noise_adapt_rate": 0.0},
        {"noise_adapt_rate": 1.5},
        {"initial_noise_db": -5.0},
        {"initial_noise_db": -101.0},
        {"calibration_ms": -1},
        {"calibration_ms": 10001},
        {"min_speech_ms": 14900, "pre_roll_ms": 200},
        {"min_silence_ms": 15001},
    ],
)
def test_invalid_config_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(VadError) as info:
        VadConfig(**kwargs)  # type: ignore[arg-type]
    assert info.value.code == "invalid_config"


def test_config_defaults_and_derived_frames() -> None:
    config = VadConfig()
    assert config.frame_samples == 320
    assert config.min_speech_frames == 10
    assert config.min_silence_frames == 20
    assert config.pre_roll_frames == 10
    assert config.calibration_frames == 15
    odd = VadConfig(frame_ms=30, min_speech_ms=200)
    assert odd.min_speech_frames == 7
    assert VadConfig(start_margin_db=6.0, stop_margin_db=6.0).stop_margin_db == 6.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.frame_ms = 10  # type: ignore[misc]


def test_segmenter_rejects_foreign_config() -> None:
    with pytest.raises(VadError) as info:
        StreamingSegmenter("nope")  # type: ignore[arg-type]
    assert info.value.code == "invalid_config"


def test_pcm16_to_float() -> None:
    """pcm16_to_float(b'abc') raises code 'invalid_audio'."""
    with pytest.raises(VadError) as info:
        pcm16_to_float(b"abc")
    assert info.value.code == "invalid_audio"
    data = bytes([0x00, 0x80, 0xFF, 0x7F, 0x00, 0x00])
    assert pcm16_to_float(data) == [-1.0, 32767 / 32768, 0.0]
    assert pcm16_to_float(bytearray(data)) == [-1.0, 32767 / 32768, 0.0]
    assert pcm16_to_float(b"") == []
    with pytest.raises(VadError):
        pcm16_to_float(123)  # type: ignore[arg-type]


def test_pcm16_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vad_segmenter, "MAX_PCM_BYTES", 2)
    with pytest.raises(VadError) as info:
        pcm16_to_float(bytes(4))
    assert info.value.code == "invalid_audio"


def test_between_margin_frame_breaks_loud_run() -> None:
    mid = 10 ** (-92.5 / 20)
    loud = const(100, 0.5)
    assert segment(zeros(300) + loud + const(20, mid) + loud + zeros(1000)) == []
    assert len(segment(zeros(300) + const(200, 0.5) + zeros(1000))) == 1


def test_between_margin_frame_breaks_quiet_run() -> None:
    mid = 10 ** (-92.5 / 20)
    body = const(400, 0.5)
    joined = zeros(300) + body + zeros(380) + const(20, mid) + zeros(380) + body + zeros(1000)
    split = zeros(300) + body + zeros(760) + body + zeros(1000)
    assert len(segment(joined)) == 1
    assert len(segment(split)) == 2


def test_noise_floor_adapts_outside_speech() -> None:
    streamer = StreamingSegmenter()
    streamer.feed(zeros(300))
    assert streamer.noise_floor_db == -100.0
    streamer.feed(const(20, 10 ** (-92.5 / 20)))
    assert streamer.noise_floor_db == pytest.approx(-99.625, abs=1e-6)
    lower = StreamingSegmenter()
    lower.feed(const(300, 0.01))
    assert lower.noise_floor_db == pytest.approx(-40.0, abs=1e-6)
    lower.feed(zeros(20))
    assert lower.noise_floor_db == pytest.approx(-43.0, abs=1e-6)
    with pytest.raises(AttributeError):
        lower.noise_floor_db = 1.0  # type: ignore[misc]


def test_speech_inside_calibration_is_a_limitation() -> None:
    streamer = StreamingSegmenter()
    assert streamer.feed(sine(1000)) == []
    assert streamer.flush() == []
    assert streamer.noise_floor_db == -10.0


def test_calibration_can_be_disabled() -> None:
    config = VadConfig(calibration_ms=0, initial_noise_db=-60.0)
    result = segment(sine(1000) + zeros(1000), config)
    assert spans(result) == [(0, 1000, False)]


def test_forced_split_with_empty_continuation_is_dropped() -> None:
    config = VadConfig(pre_roll_ms=0, max_segment_ms=1000)
    result = segment(zeros(300) + sine(600) + zeros(1000), config)
    assert spans(result) == [(300, 1300, True)]
    assert len(result[0].samples) == 16000


def test_pre_roll_of_next_segment_does_not_overlap_previous() -> None:
    signal = zeros(1000) + sine(1000) + zeros(600) + sine(1000) + zeros(1000)
    assert spans(segment(signal)) == [(800, 2000, False), (2400, 3600, False)]


def test_partial_frame_and_pending_run_are_dropped() -> None:
    partial = sine(100)[:100]
    result = segment(zeros(1000) + sine(1000) + partial)
    assert spans(result) == [(800, 2000, False)]
    assert len(result[0].samples) == 19200
    assert segment(zeros(1000) + sine(100)) == []


def test_other_sample_rate_and_frame_size() -> None:
    config = VadConfig(sample_rate=8000, frame_ms=30)
    signal = zeros(1000, 8000) + sine(600, rate=8000) + zeros(1000, 8000)
    result = segment(signal, config)
    assert len(result) == 1
    seg = result[0]
    assert seg.start_ms % 30 == 0
    assert seg.end_ms % 30 == 0
    assert 700 <= seg.start_ms <= 1000
    assert 1500 <= seg.end_ms <= 1650
    assert len(seg.samples) == (seg.end_ms - seg.start_ms) * 8


def test_segment_is_immutable() -> None:
    seg = segment(zeros(1000) + sine(1000) + zeros(1000))[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        seg.start_ms = 0  # type: ignore[misc]


def test_error_message_is_capped_and_logger_is_quiet() -> None:
    assert len(str(VadError("closed", "x" * 500))) == 200
    handlers = logging.getLogger("vad_segmenter").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
    assert vad_segmenter.__version__ == "0.1.0"
