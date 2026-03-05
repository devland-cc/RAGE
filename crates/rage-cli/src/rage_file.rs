use std::io::Write;
use std::path::Path;
use std::time::SystemTime;

use anyhow::Result;
use rage_core::DeepAnalysis;

/// Format a SystemTime as an ISO 8601 string (UTC, second precision).
fn format_timestamp(time: SystemTime) -> String {
    let dur = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();

    // Simple UTC conversion (no leap seconds, good enough for timestamps)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since 1970-01-01 to Y-M-D
    let mut y = 1970i32;
    let mut remaining_days = days as i32;

    loop {
        let year_days = if is_leap(y) { 366 } else { 365 };
        if remaining_days < year_days {
            break;
        }
        remaining_days -= year_days;
        y += 1;
    }

    let month_days = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut m = 0;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining_days < md {
            m = i;
            break;
        }
        remaining_days -= md;
    }

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m + 1,
        remaining_days + 1,
        hours,
        minutes,
        seconds
    )
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Write a DeepAnalysis to a .rage file.
pub fn write_rage_file(analysis: &DeepAnalysis, path: &Path) -> Result<()> {
    let mut f = std::fs::File::create(path)?;

    // Header
    writeln!(f, "# RAGE Deep Analysis v0.0.1")?;
    writeln!(f, "# Source: {}", analysis.source)?;
    writeln!(f, "# Duration: {:.1}s", analysis.duration_secs)?;
    writeln!(f, "# Generated: {}", format_timestamp(SystemTime::now()))?;
    writeln!(f)?;

    // Summary
    writeln!(f, "[SUMMARY]")?;
    writeln!(
        f,
        "dominant_bpm={}",
        analysis.summary.dominant_bpm.round() as i32
    )?;
    writeln!(f, "dominant_key={}", analysis.summary.dominant_key)?;
    writeln!(f, "avg_valence={:+.3}", analysis.summary.avg_valence)?;
    writeln!(f, "avg_arousal={:+.3}", analysis.summary.avg_arousal)?;
    for (i, mood) in analysis.summary.top_moods.iter().enumerate() {
        writeln!(
            f,
            "mood_{}={}:{:.3}",
            i + 1,
            mood.tag.name(),
            mood.probability
        )?;
    }
    writeln!(f)?;

    // Beats
    writeln!(f, "[BEATS]")?;
    writeln!(f, "# time|bpm|key")?;
    for beat in &analysis.beats {
        writeln!(f, "{:.3}|{:.1}|{}", beat.time_secs, beat.bpm, beat.key)?;
    }
    writeln!(f)?;

    // Emotions
    writeln!(f, "[EMOTIONS]")?;
    writeln!(f, "# time|valence|arousal|moods")?;
    for seg in &analysis.segments {
        let moods_str: Vec<String> = seg
            .mood_tags
            .iter()
            .map(|m| format!("{}:{:.2}", m.tag.name(), m.probability))
            .collect();
        writeln!(
            f,
            "{:.3}|{:+.3}|{:+.3}|{}",
            seg.time_secs,
            seg.valence,
            seg.arousal,
            moods_str.join(",")
        )?;
    }

    Ok(())
}

/// Print a summary of the deep analysis to stdout.
pub fn print_deep_summary(analysis: &DeepAnalysis) {
    println!("  {}", analysis.source);
    println!("  Duration: {:.1}s", analysis.duration_secs);
    println!();

    println!(
        "  Dominant BPM: {}",
        analysis.summary.dominant_bpm.round() as i32
    );
    println!("  Dominant Key: {}", analysis.summary.dominant_key);
    println!();

    println!("  Avg Valence: {:+.3}", analysis.summary.avg_valence);
    println!("  Avg Arousal: {:+.3}", analysis.summary.avg_arousal);
    println!();

    if !analysis.summary.top_moods.is_empty() {
        println!("  Top Moods:");
        for (i, mood) in analysis.summary.top_moods.iter().enumerate() {
            println!(
                "    {:2}. {:<16} {:.3}",
                i + 1,
                mood.tag.name(),
                mood.probability
            );
        }
        println!();
    }

    println!("  Beats: {}", analysis.beats.len());
    println!("  Segments: {}", analysis.segments.len());
    println!();
}
