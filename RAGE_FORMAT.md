# .RAGE File Format Specification (v0.0.1)

> **Work in progress** — This format is in its early stages and is expected to evolve quickly. If you're thinking about building on it or contributing, please reach out at **rage@devland.cc** with suggestions, questions, or ideas.

The `.rage` file is a plain-text format produced by `rage deep`. It contains a complete temporal analysis of a music track: per-beat BPM and key detection, segmented emotion analysis, and summary statistics.

## Structure

A `.rage` file has four parts: a header, and three sections (`[SUMMARY]`, `[BEATS]`, `[EMOTIONS]`).

Lines starting with `#` are comments. Sections are introduced by `[SECTION_NAME]` on its own line. Fields within sections use `=` for key-value pairs or `|` (pipe) as a delimiter for tabular data.

## Header

```
# RAGE Deep Analysis v0.0.1
# Source: song.mp3
# Duration: 240.1s
# Generated: 2026-03-05T14:30:00Z
```

- **v0.0.1**: Format version.
- **Source**: Original filename.
- **Duration**: Total audio duration in seconds.
- **Generated**: ISO 8601 UTC timestamp.

## [SUMMARY]

```
[SUMMARY]
dominant_bpm=120
dominant_key=G major
avg_valence=+0.234
avg_arousal=-0.112
mood_1=energetic:0.523
mood_2=dark:0.412
mood_3=powerful:0.389
mood_4=epic:0.301
mood_5=film:0.278
```

| Field | Type | Description |
|-------|------|-------------|
| `dominant_bpm` | integer | Most frequent rounded BPM across all beats |
| `dominant_key` | string | Most frequent key across all beats (e.g. `G major`, `F# minor`) |
| `avg_valence` | signed float | Mean valence across all segments, range [-1, +1] |
| `avg_arousal` | signed float | Mean arousal across all segments, range [-1, +1] |
| `mood_N` | tag:probability | Top mood tags ranked by average probability (N = 1..5) |

## [BEATS]

```
[BEATS]
# time|bpm|key
0.371|120.2|G major
0.873|119.8|G major
1.375|120.1|G major
```

Pipe-delimited, one row per detected beat.

| Column | Type | Description |
|--------|------|-------------|
| `time` | float (3 decimals) | Beat timestamp in seconds from start of track |
| `bpm` | float (1 decimal) | Local BPM computed from the interval to the next beat |
| `key` | string | Musical key detected from chroma features around this beat |

**Key format**: Pitch class + mode, e.g. `C major`, `F# minor`, `A# major`. Pitch classes use sharps (no flats): C, C#, D, D#, E, F, F#, G, G#, A, A#, B.

**BPM calculation**: Local BPM is `60 / (next_beat_time - this_beat_time)`. The last beat inherits the BPM of the preceding beat.

**Key detection**: Uses Krumhansl-Kessler key profiles correlated against chroma features in a window of +/- half a beat period around each beat position.

## [EMOTIONS]

```
[EMOTIONS]
# time|valence|arousal|moods
0.000|+0.312|-0.045|energetic:0.62,dark:0.41,powerful:0.38,film:0.29,epic:0.25
20.000|+0.189|-0.201|dark:0.55,deep:0.42,emotional:0.38,melancholic:0.31,drama:0.27
```

Pipe-delimited, one row per time segment. The `moods` column uses commas to separate tag:probability pairs.

| Column | Type | Description |
|--------|------|-------------|
| `time` | float (3 decimals) | Segment start time in seconds |
| `valence` | signed float | Valence score [-1, +1] for this segment |
| `arousal` | signed float | Arousal score [-1, +1] for this segment |
| `moods` | comma-separated | Top mood tags with probabilities (tag:prob format) |

**Segmentation**: Audio is split into fixed-length segments (default 20 seconds). The final segment is processed as-is if it is at least 5 seconds long; otherwise it is merged with the previous segment.

**Mood tags**: Drawn from the 56-tag MTG-Jamendo taxonomy. Only the top-k tags (default 5) are included per segment.

## Algorithms

### Beat tracking

1. Full-length STFT magnitude is computed (2048-point FFT, 512 hop).
2. Onset strength is derived via spectral flux (half-wave rectified frame-to-frame magnitude increase).
3. Global tempo is estimated via autocorrelation of the onset strength signal, searching 60-200 BPM with a Gaussian prior centered at 120 BPM.
4. Beat positions are tracked using dynamic programming that maximizes onset strength while maintaining regular spacing (simplified Ellis 2007).

### Key detection

1. Chroma features (12 pitch classes) are computed from the STFT magnitude.
2. For each beat, the mean chroma vector is computed over a window of +/- half a beat period.
3. Pearson correlation is computed against all 24 Krumhansl-Kessler key profiles (12 major + 12 minor).
4. The key with the highest correlation is selected.

### Emotion analysis

1. Audio is split into segments (default 20s).
2. Each segment is processed through the full feature extraction pipeline (mel spectrogram, MFCCs, chroma, spectral features, summary vector).
3. The MoodTagger CNN predicts 56 mood tag probabilities.
4. The ValenceArousal CNN+MLP predicts continuous valence and arousal values.

## Parsing

The format is designed for easy parsing in any language:

1. Read lines, skip lines starting with `#` or empty lines.
2. Track the current section via `[SECTION_NAME]` lines.
3. In `[SUMMARY]`: split on `=` for key-value pairs; mood values split further on `:`.
4. In `[BEATS]` and `[EMOTIONS]`: split on `|` for columns; mood lists split on `,` then `:`.
