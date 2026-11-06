# Adaptive energy VAD with hysteresis in pure Python

Status: accepted (generated, pending human review)

## Context

Live transcription needs audio cut into speech segments. webrtcvad is a C extension with fixed aggressiveness and Silero VAD needs torch or onnxruntime. Many pipelines only need a small, dependency-free, fully testable detector with a streaming API.

## Decision

Implement a frame-based energy detector on the standard library only. A calibration window sets the noise floor, then an exponential moving average tracks it outside speech. Separate start and stop margins give hysteresis. Pre-roll sits in a bounded deque and a forced split caps segment length. Chunking never changes the result because the state machine only sees whole frames. Timestamps come from frame counts, and inputs are validated before any state change.

## Consequences

The library is portable and deterministic and needs no native code, but it is less robust than neural VAD on non-stationary noise. Speech inside the calibration window raises the floor, which is a documented limitation. Pure-Python per-sample loops limit throughput to roughly real-time streams. Public errors are a single VadError with stable codes.
