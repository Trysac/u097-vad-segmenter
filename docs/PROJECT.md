# Project record

## Objective

Provide a pure-Python adaptive energy-based voice activity detector that turns streamed audio samples into speech segments for live transcription, with hysteresis, pre-roll and a streaming API.

## Scope

- In scope: the public API documented in [README.md](../README.md): `VadConfig`, `StreamingSegmenter`, `segment`, `Segment`, `frame_energy_db`, `pcm16_to_float`, `VadError`, typed and tested.
- Out of scope: neural VAD, resampling, audio capture, numpy, and any runtime dependency without an accepted ADR.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | 302269 / 64884 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.5198 |

## Factory run u097 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | claude-sonnet-5-5 | yes | 2350 | 53908 | 0 | 4246 | 0.2723 |
| audit | claude-sonnet-5-5 | no | 12 | 4186 | 27173 | 120722 | 0.1219 |
| fix-1 | claude-sonnet-5-5 | no | 6 | 2161 | 8530 | 40877 | 0.0470 |
| docs | claude-haiku-5-5 | no | 134056 | 4629 | 0 | 0 | 0.0786 |

Total: $0.5198. Prices: Anthropic list prices.
