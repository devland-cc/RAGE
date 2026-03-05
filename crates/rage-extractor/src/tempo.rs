use ndarray::Array1;

/// Estimate tempo (BPM) from onset strength via autocorrelation.
///
/// Searches lags corresponding to 60–200 BPM range.
/// Applies a Gaussian prior centered at 120 BPM (sigma=40) to bias toward
/// plausible tempos.
///
/// `fps` is the frame rate of the onset strength signal (sample_rate / hop_length).
pub fn estimate_tempo(onset: &Array1<f32>, fps: f32) -> f32 {
    let n = onset.len();
    if n < 2 {
        return 120.0;
    }

    // Lag range for 60–200 BPM
    let min_lag = (fps * 60.0 / 200.0).ceil() as usize; // fastest tempo → shortest lag
    let max_lag = (fps * 60.0 / 60.0).floor() as usize; // slowest tempo → longest lag
    let max_lag = max_lag.min(n - 1);

    if min_lag >= max_lag {
        return 120.0;
    }

    let prior_center = 120.0f32;
    let prior_sigma = 40.0f32;

    let mut best_lag = min_lag;
    let mut best_score = f32::NEG_INFINITY;

    for lag in min_lag..=max_lag {
        // Autocorrelation at this lag
        let mut corr = 0.0f32;
        for t in lag..n {
            corr += onset[t] * onset[t - lag];
        }
        corr /= (n - lag) as f32;

        // Gaussian prior on BPM
        let bpm = fps * 60.0 / lag as f32;
        let prior = (-(bpm - prior_center).powi(2) / (2.0 * prior_sigma.powi(2))).exp();

        let score = corr * prior;
        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    fps * 60.0 / best_lag as f32
}

/// Track beat positions using dynamic programming (simplified Ellis 2007).
///
/// Given an onset strength signal and estimated tempo, finds beat positions
/// that maximize onset strength while maintaining regular spacing.
///
/// Returns frame indices of detected beats, sorted chronologically.
///
/// `fps` is the frame rate of the onset strength signal (sample_rate / hop_length).
pub fn track_beats(onset: &Array1<f32>, tempo_bpm: f32, fps: f32) -> Vec<usize> {
    let n = onset.len();
    if n < 2 {
        return vec![];
    }

    // Expected beat period in frames
    let period = (fps * 60.0 / tempo_bpm).round() as usize;
    if period == 0 {
        return vec![];
    }

    // Allow ~12.5% tempo variation
    let delta = (period / 8).max(1);
    let alpha = 4.0f32; // Transition weight

    // Forward pass: compute scores
    let mut score = vec![0.0f32; n];
    let mut backptr = vec![0usize; n];

    // Initialize first frames
    for t in 0..n.min(period + delta) {
        score[t] = onset[t];
        backptr[t] = t; // self-pointer = start of chain
    }

    for t in (period.saturating_sub(delta))..n {
        let search_start = t.saturating_sub(period + delta);
        let search_end = t.saturating_sub(period.saturating_sub(delta)).min(t);

        if search_start >= search_end {
            score[t] = onset[t];
            backptr[t] = t;
            continue;
        }

        let mut best_prev = search_start;
        let mut best_prev_score = score[search_start];

        for prev in search_start..search_end {
            if score[prev] > best_prev_score {
                best_prev_score = score[prev];
                best_prev = prev;
            }
        }

        let transition_score = onset[t] + alpha * best_prev_score;
        if transition_score > score[t] {
            score[t] = transition_score;
            backptr[t] = best_prev;
        }
    }

    // Find the best endpoint (search in last beat period)
    let search_from = n.saturating_sub(period);
    let mut best_end = search_from;
    let mut best_end_score = score[search_from];
    for t in search_from..n {
        if score[t] > best_end_score {
            best_end_score = score[t];
            best_end = t;
        }
    }

    // Backtrack to recover beat sequence
    let mut beats = Vec::new();
    let mut t = best_end;
    beats.push(t);

    loop {
        let prev = backptr[t];
        if prev == t || prev >= t {
            break;
        }
        beats.push(prev);
        t = prev;
    }

    beats.reverse();
    beats
}

/// Compute local BPM for each beat from consecutive inter-beat intervals.
///
/// Returns a vector of BPM values, one per beat. The last beat gets the same
/// BPM as the second-to-last.
///
/// `fps` is the frame rate (sample_rate / hop_length).
pub fn local_bpm(beats: &[usize], fps: f32) -> Vec<f32> {
    if beats.is_empty() {
        return vec![];
    }
    if beats.len() == 1 {
        return vec![0.0];
    }

    let mut bpms = Vec::with_capacity(beats.len());

    for i in 0..beats.len() - 1 {
        let interval_frames = beats[i + 1] as f32 - beats[i] as f32;
        if interval_frames > 0.0 {
            bpms.push(fps * 60.0 / interval_frames);
        } else {
            bpms.push(0.0);
        }
    }

    // Last beat gets same BPM as previous
    let last_bpm = *bpms.last().unwrap_or(&0.0);
    bpms.push(last_bpm);

    bpms
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_click_track(bpm: f32, fps: f32, duration_secs: f32) -> Array1<f32> {
        let n_frames = (fps * duration_secs) as usize;
        let period_frames = (fps * 60.0 / bpm).round() as usize;
        let mut onset = Array1::<f32>::zeros(n_frames);

        let mut t = 0;
        while t < n_frames {
            onset[t] = 1.0;
            t += period_frames;
        }

        onset
    }

    #[test]
    fn estimate_tempo_120bpm() {
        let fps = 22050.0 / 512.0; // ~43.07 fps
        let onset = make_click_track(120.0, fps, 10.0);
        let bpm = estimate_tempo(&onset, fps);
        assert!(
            (bpm - 120.0).abs() < 5.0,
            "expected ~120 BPM, got {bpm:.1}"
        );
    }

    #[test]
    fn estimate_tempo_90bpm() {
        let fps = 22050.0 / 512.0;
        let onset = make_click_track(90.0, fps, 10.0);
        let bpm = estimate_tempo(&onset, fps);
        assert!(
            (bpm - 90.0).abs() < 5.0,
            "expected ~90 BPM, got {bpm:.1}"
        );
    }

    #[test]
    fn estimate_tempo_150bpm() {
        let fps = 22050.0 / 512.0;
        let onset = make_click_track(150.0, fps, 10.0);
        let bpm = estimate_tempo(&onset, fps);
        assert!(
            (bpm - 150.0).abs() < 5.0,
            "expected ~150 BPM, got {bpm:.1}"
        );
    }

    #[test]
    fn track_beats_120bpm() {
        let fps = 22050.0 / 512.0;
        let onset = make_click_track(120.0, fps, 5.0);
        let bpm = estimate_tempo(&onset, fps);
        let beats = track_beats(&onset, bpm, fps);

        // At 120 BPM for 5 seconds, expect ~10 beats
        assert!(
            beats.len() >= 7 && beats.len() <= 13,
            "expected ~10 beats, got {}",
            beats.len()
        );

        // Beats should be roughly evenly spaced
        let expected_period = (fps * 60.0 / 120.0).round() as usize;
        for i in 1..beats.len() {
            let interval = beats[i] - beats[i - 1];
            let error = (interval as f32 - expected_period as f32).abs();
            assert!(
                error < expected_period as f32 * 0.2,
                "beat interval {} deviates too much from expected {}",
                interval,
                expected_period
            );
        }
    }

    #[test]
    fn local_bpm_calculation() {
        let fps = 22050.0 / 512.0;
        // Simulate beats at exactly 120 BPM
        let period = (fps * 60.0 / 120.0f32).round() as usize;
        let beats: Vec<usize> = (0..10).map(|i| i * period).collect();

        let bpms = local_bpm(&beats, fps);
        assert_eq!(bpms.len(), beats.len());

        for &bpm in &bpms {
            assert!(
                (bpm - 120.0).abs() < 5.0,
                "expected ~120 BPM, got {bpm:.1}"
            );
        }
    }

    #[test]
    fn local_bpm_empty() {
        assert!(local_bpm(&[], 43.0).is_empty());
    }

    #[test]
    fn local_bpm_single() {
        assert_eq!(local_bpm(&[0], 43.0), vec![0.0]);
    }
}
