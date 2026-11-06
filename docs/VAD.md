# Voice activity detection

`vad_segmenter` is a pure-Python adaptive energy detector. It turns streamed audio samples
into speech segments (with the audio) for live transcription.

## Energy convention

The energy of a frame is `20 * log10(rms)` over the frame, floored at -100.0 dB. Full scale is a
square wave at 1.0, so a full-scale sine reads -3.01 dBFS. This is not the AES17 convention.
Samples must be finite numbers; values outside [-1, 1] are clipped.

## Algorithm

1. Frames are consecutive non-overlapping blocks of `sample_rate * frame_ms / 1000` samples;
   leftover samples are buffered between `feed` calls and a partial final frame is dropped by
   `flush`.
2. A frame is loud when its energy is above `noise_floor + start_margin_db` and quiet when it is
   below `noise_floor + stop_margin_db`. A frame between the margins breaks both runs.
3. The first `calibration_ms` of a stream never start speech and set the floor to their mean
   energy. Afterwards the floor is an exponential moving average updated only on frames outside
   speech that are not loud, clamped to [-100, -10] dB. Speech inside the calibration window is a
   documented limitation: it raises the floor.
4. Speech starts after `min_speech_ms` of consecutive loud frames (back-dated to the first) and
   ends after `min_silence_ms` of consecutive quiet frames; the segment ends at the end of the
   last loud frame.
5. A segment includes `pre_roll_ms` of earlier audio. A segment reaching `max_segment_ms` is cut
   with `forced_split=True`; the continuation has no pre-roll. Memory stays within
   `max_segment_ms` plus pre-roll.

`peak_db` and `mean_db` of a `Segment` are the maximum and the mean of the per-frame dBFS values
of all frames in the segment, pre-roll included.

## Limits

Inputs are capped by `MAX_FEED_SAMPLES`, `MAX_FRAME_SAMPLES`, `MAX_SEGMENT_INPUT_SAMPLES` and
`MAX_PCM_BYTES`; violations raise `VadError` with code `invalid_audio`. Out of scope: neural VAD,
resampling, audio capture and numpy.
