# vad-segmenter

[![PyPI version](https://img.shields.io/pypi/v/trysac-vad-segmenter)](https://pypi.org/project/trysac-vad-segmenter/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-vad-segmenter)](LICENSE)
[![CI](https://github.com/Trysac/u097-vad-segmenter/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u097-vad-segmenter/actions/workflows/ci.yml)

Pure-Python adaptive energy-based voice activity detection that turns streamed audio samples into speech segments for live transcription.

## Problem

Live transcription needs audio cut into speech segments: too short and words are clipped, too long and latency grows. webrtcvad is a C extension with fixed aggressiveness and Silero VAD needs torch or onnxruntime. This package is an adaptive energy detector with hysteresis, pre-roll and a streaming API, written in pure Python with no runtime dependencies.

## Installation

```sh
pip install trysac-vad-segmenter
# or
uv add trysac-vad-segmenter
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import math
from vad_segmenter import segment

silence = [0.0] * 16000
tone = [0.5 * math.sin(2 * math.pi * 440 * n / 16000) for n in range(16000)]
for seg in segment(silence + tone + silence):
    print(seg.start_ms, seg.end_ms, seg.forced_split)  # 800 2000 False
```

The install name is `trysac-vad-segmenter`; the import name is `vad_segmenter`.

## Public API

- `VadConfig(...)`: frozen dataclass, validated on construction. Fields: `sample_rate` (8000, 16000, 32000 or 48000, default 16000), `frame_ms` (10, 20 or 30, default 20), `start_margin_db` (9.0), `stop_margin_db` (6.0, must not exceed start), `min_speech_ms` (200), `min_silence_ms` (400), `pre_roll_ms` (200, range 0-1000), `max_segment_ms` (15000, range 1000-120000), `noise_adapt_rate` (0.05, in (0, 1]), `initial_noise_db` (-60.0, range -100 to -10), `calibration_ms` (300, range 0-10000). Durations are rounded up to whole frames. `min_speech_ms + pre_roll_ms` must not exceed `max_segment_ms`, and `min_silence_ms` must not exceed `max_segment_ms`.
- `StreamingSegmenter(config=None)`: `feed(samples) -> list[Segment]` consumes samples and returns completed segments; `flush() -> list[Segment]` ends the stream and is terminal; read-only property `noise_floor_db`.
- `segment(samples, config=None) -> list[Segment]`: one-shot equivalent of feeding everything and flushing.
- `Segment`: frozen dataclass with `start_ms`, `end_ms`, `samples` (tuple of floats, pre-roll included), `forced_split`, `peak_db`, `mean_db` (max and mean of per-frame dBFS over the frames in the segment).
- `frame_energy_db(frame) -> float`: dBFS energy of one frame, floored at -100.0.
- `pcm16_to_float(data: bytes) -> list[float]`: little-endian signed 16-bit PCM divided by 32768.
- `VadError`: attribute `code` is `invalid_config`, `invalid_audio` or `closed`.

Energy is dBFS = 20 * log10(rms), floored at -100.0. Full scale is a square wave at 1.0, so a full-scale sine reads -3.01 dBFS. This is not AES17.

### Errors

- `invalid_config`: a field is out of range, has the wrong type, or breaks a cross-field rule.
- `invalid_audio`: a sample is not a finite number, a frame is empty, an input exceeds the limits, or PCM data has an odd byte length.
- `closed`: `feed` or `flush` was called after `flush`.

### Limits

- `MAX_FRAME_SAMPLES` = 1,000,000 samples per frame.
- `MAX_FEED_SAMPLES` = 10,000,000 samples per `feed` call.
- `MAX_SEGMENT_INPUT_SAMPLES` = 100,000,000 samples per `segment` call.
- `MAX_PCM_BYTES` = 20,000,000 bytes per `pcm16_to_float` call.
- Samples outside [-1, 1] are clipped.

### Limitations

- Speech inside the calibration window (the first `calibration_ms`) is not detected; it raises the noise floor.
- `flush()` drops a partial final frame.
- No neural VAD, resampling or audio capture.

## Why not webrtcvad

webrtcvad is a C extension with fixed aggressiveness. This package is pure Python, adapts its noise floor to the stream, and returns segments with their audio.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md) (threat model: [docs/SECURITY.md](docs/SECURITY.md))
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez
