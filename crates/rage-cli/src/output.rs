use rage_core::types::EmotionResult;

/// Print emotion analysis results as a human-readable table.
pub fn print_emotion_table(result: &EmotionResult, top_k: usize) {
    println!("  {}", result.source);
    println!();

    if let Some(va) = &result.valence_arousal {
        let v_label = if va.valence > 0.2 {
            "positive"
        } else if va.valence < -0.2 {
            "negative"
        } else {
            "neutral"
        };
        let a_label = if va.arousal > 0.2 {
            "energetic"
        } else if va.arousal < -0.2 {
            "calm"
        } else {
            "moderate"
        };
        println!(
            "  Valence: {:+.3}  ({v_label})",
            va.valence,
        );
        println!(
            "  Arousal: {:+.3}  ({a_label})",
            va.arousal,
        );
        println!();
    }

    let n = top_k.min(result.mood_tags.len());
    if n > 0 {
        println!("  Top Mood Tags:");
        for (i, pred) in result.mood_tags.iter().take(n).enumerate() {
            println!(
                "    {:2}. {:<16} {:.3}",
                i + 1,
                pred.tag.name(),
                pred.probability,
            );
        }
        println!();
    }
}

/// Print emotion analysis results as JSON.
pub fn print_emotion_json(result: &EmotionResult) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(result)?);
    Ok(())
}
