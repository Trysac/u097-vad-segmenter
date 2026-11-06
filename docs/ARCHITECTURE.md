# Architecture

## Purpose

`trysac-vad-segmenter` is a dependency-free Python library (import name `vad_segmenter`) that turns streamed audio samples into speech segments using an adaptive energy detector.

## Shape

```text
caller code --> vad_segmenter (public API in __init__.py)
                    |
                    +-- StreamingSegmenter: frame buffer, noise floor, speech state machine
                    +-- pure helpers: frame_energy_db, pcm16_to_float, segment
```

- **Public API**: the six names in `__all__` of `src/vad_segmenter/__init__.py` plus `__version__`.
- **Algorithm**: frames are consecutive non-overlapping blocks. Each frame is classified as loud, quiet or neither against a noise floor. Speech starts after `min_speech_ms` of loud frames and ends after `min_silence_ms` of quiet frames. A pre-roll deque holds audio before the start. A segment reaching `max_segment_ms` is cut with `forced_split=True`.
- **Errors**: all errors are `VadError` with a stable `code` attribute.
- **Side effects**: none beyond a module logger (`logging.getLogger(__name__)`) with a NullHandler that emits debug messages per segment.

## Quality contract

The gates are run by `scripts/check.py`. See [TESTING.md](TESTING.md).

## Decisions

Architecture decisions are recorded in [decisions/](decisions/) as numbered ADRs.
