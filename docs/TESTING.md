# Testing

## Running

```sh
uv run pytest --cov=vad_segmenter   # tests with coverage report
uv run python scripts/check.py      # every gate, as CI and the factory run it
```

## Existing tests

`tests/test_vad_segmenter.py` tests the public API through `import vad_segmenter` and covers:

- the pre-roll and segment boundaries of a 440 Hz tone between silences (`test_single_segment_with_pre_roll`);
- short bursts below `min_speech_ms` and gaps below `min_silence_ms` (`test_short_burst_and_short_gap`);
- equal output for different chunk sizes (`test_chunking_does_not_change_segments`);
- the forced split at `max_segment_ms` and the continuation without pre-roll;
- noise calibration of the floor on synthetic random noise;
- `frame_energy_db` values for a full-scale sine and an all-zero frame;
- `flush`, empty input, `closed` and `invalid_audio` errors;
- `VadConfig` validation and `invalid_config` errors;
- `pcm16_to_float` conversion and its `invalid_audio` error.

No audio files are used; signals are synthesized in the tests.

## Policy

- Branch coverage of `src/vad_segmenter` must stay at or above 80 %; the gate fails below it.
- Tests assert return values, raised exceptions and their codes.
- Each bug fix starts with a failing test that reproduces it.
- Tests are deterministic: no network, no wall clock, no dependence on test order.

## Layout

`tests/test_vad_segmenter.py` covers the module `src/vad_segmenter/__init__.py`.
