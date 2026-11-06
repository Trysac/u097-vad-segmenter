# Security notes

## Threat model

`trysac-vad-segmenter` is a pure library without I/O, network access or file access.

### Defended

- **Non-finite samples**: NaN and infinite values are rejected with `VadError` code `invalid_audio`. Values outside [-1, 1] are clipped.
- **Oversized input**: `MAX_FRAME_SAMPLES` (1,000,000), `MAX_FEED_SAMPLES` (10,000,000), `MAX_SEGMENT_INPUT_SAMPLES` (100,000,000) and `MAX_PCM_BYTES` (20,000,000) bound each call.
- **Unbounded memory in a stream**: segment memory is bounded by `max_segment_ms` plus `pre_roll_ms`, because a long segment is force-split.
- **Invalid configuration**: every `VadConfig` field is range-checked and cross-field rules are enforced.
- **Unsafe deserialization**: PCM input is parsed with `struct.unpack` of a fixed format; no `eval`, `exec` or pickle is used.

### Not defended

- **Adversarial audio**: an attacker can craft audio that evades or triggers detection; the detector is energy-based and makes no robustness claim.
- **Noise floor manipulation**: speech inside the calibration window raises the floor and can hide later speech.
- **Resource use within limits**: inputs near the limits still cost CPU time, since processing is pure Python.
- **Supply chain of the environment**: the library has no runtime dependencies, but development tooling is outside its control.

## Publishing

Releases are published to PyPI with trusted publishing (OIDC) when available, never with a long-lived token stored in the repository.

To report a vulnerability, see [SECURITY.md](../SECURITY.md).
