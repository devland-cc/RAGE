use std::path::Path;

use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use rage_core::types::{EmotionResult, MoodPrediction, ValenceArousal};
use rage_core::{MoodTag, RageError};
use rage_extractor::FeatureSet;

/// Expected number of time frames for the ONNX models (30s at 22050Hz / 512 hop).
const MODEL_N_FRAMES: usize = 1292;
/// Number of mel bins.
const MODEL_N_MELS: usize = 128;
/// Number of mood tags the model outputs.
const MODEL_N_TAGS: usize = 56;
/// Summary vector dimensionality.
const MODEL_SUMMARY_DIM: usize = 294;

/// ONNX-based classifier for music emotion recognition.
///
/// Holds two ONNX Runtime sessions:
/// - MoodTagger: mel spectrogram -> 56 mood tag logits
/// - ValenceArousal: mel spectrogram + summary vector -> (valence, arousal)
pub struct Classifier {
    mood_session: Session,
    va_session: Session,
}

impl Classifier {
    /// Load both ONNX models from the given directory.
    ///
    /// Expects `mood_tagger.onnx` and `valence_arousal.onnx` in `models_dir`.
    pub fn from_dir(models_dir: &Path) -> Result<Self, RageError> {
        let mood_path = models_dir.join("mood_tagger.onnx");
        let va_path = models_dir.join("valence_arousal.onnx");

        if !mood_path.exists() {
            return Err(RageError::Inference(format!(
                "mood tagger model not found: {}",
                mood_path.display()
            )));
        }
        if !va_path.exists() {
            return Err(RageError::Inference(format!(
                "valence-arousal model not found: {}",
                va_path.display()
            )));
        }

        let mood_session = Session::builder()
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.commit_from_file(&mood_path))
            .map_err(|e| RageError::Inference(format!("failed to load mood tagger: {e}")))?;

        let va_session = Session::builder()
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.commit_from_file(&va_path))
            .map_err(|e| RageError::Inference(format!("failed to load V-A model: {e}")))?;

        Ok(Self {
            mood_session,
            va_session,
        })
    }

    /// Run both models on extracted features and return the emotion analysis.
    pub fn classify(
        &mut self,
        features: &FeatureSet,
        source: &str,
    ) -> Result<EmotionResult, RageError> {
        let mel_input = prepare_mel_tensor(&features.log_mel_spectrogram);
        let mood_tags = self.run_mood_tagger(&mel_input)?;
        let valence_arousal =
            self.run_va_model(&mel_input, &features.summary_vector)?;

        Ok(EmotionResult {
            source: source.to_string(),
            valence_arousal: Some(valence_arousal),
            mood_tags,
        })
    }

    fn run_mood_tagger(
        &mut self,
        mel: &Array4<f32>,
    ) -> Result<Vec<MoodPrediction>, RageError> {
        let mel_ref = TensorRef::from_array_view(mel)
            .map_err(|e| RageError::Inference(format!("mood tagger input error: {e}")))?;

        let inputs = ort::inputs![
            "mel_spectrogram" => mel_ref,
        ];

        let outputs = self
            .mood_session
            .run(inputs)
            .map_err(|e| RageError::Inference(format!("mood tagger inference error: {e}")))?;

        let (_, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| RageError::Inference(format!("mood tagger output error: {e}")))?;

        if logits_data.len() < MODEL_N_TAGS {
            return Err(RageError::Inference(format!(
                "mood tagger output has {} elements, expected {}",
                logits_data.len(),
                MODEL_N_TAGS
            )));
        }

        let mut predictions: Vec<MoodPrediction> = MoodTag::ALL
            .iter()
            .enumerate()
            .map(|(i, &tag)| MoodPrediction {
                tag,
                probability: sigmoid(logits_data[i]),
            })
            .collect();

        predictions.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(predictions)
    }

    fn run_va_model(
        &mut self,
        mel: &Array4<f32>,
        summary_vector: &[f32],
    ) -> Result<ValenceArousal, RageError> {
        if summary_vector.len() != MODEL_SUMMARY_DIM {
            return Err(RageError::Inference(format!(
                "summary vector has {} dims, expected {}",
                summary_vector.len(),
                MODEL_SUMMARY_DIM
            )));
        }

        let mel_ref = TensorRef::from_array_view(mel)
            .map_err(|e| RageError::Inference(format!("V-A mel input error: {e}")))?;

        let summary_array =
            ndarray::Array2::from_shape_vec((1, MODEL_SUMMARY_DIM), summary_vector.to_vec())
                .map_err(|e| {
                    RageError::Inference(format!("summary vector reshape error: {e}"))
                })?;

        let summary_ref = TensorRef::from_array_view(&summary_array)
            .map_err(|e| RageError::Inference(format!("V-A summary input error: {e}")))?;

        let inputs = ort::inputs![
            "mel_spectrogram" => mel_ref,
            "summary_vector" => summary_ref,
        ];

        let outputs = self
            .va_session
            .run(inputs)
            .map_err(|e| RageError::Inference(format!("V-A model inference error: {e}")))?;

        let (_, va_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| RageError::Inference(format!("V-A model output error: {e}")))?;

        if va_data.len() < 2 {
            return Err(RageError::Inference(format!(
                "V-A model output has {} elements, expected 2",
                va_data.len()
            )));
        }

        Ok(ValenceArousal {
            valence: va_data[0].clamp(-1.0, 1.0),
            arousal: va_data[1].clamp(-1.0, 1.0),
        })
    }
}

/// Prepare the mel spectrogram tensor for ONNX input.
///
/// Takes a [n_mels, n_frames] array and reshapes to [1, 1, 128, 1292],
/// padding with zeros or trimming as needed.
fn prepare_mel_tensor(log_mel: &ndarray::Array2<f32>) -> Array4<f32> {
    let (n_mels, n_frames) = (log_mel.shape()[0], log_mel.shape()[1]);

    let mut tensor = Array4::<f32>::zeros((1, 1, MODEL_N_MELS, MODEL_N_FRAMES));

    let mels_to_copy = n_mels.min(MODEL_N_MELS);
    let frames_to_copy = n_frames.min(MODEL_N_FRAMES);

    for m in 0..mels_to_copy {
        for f in 0..frames_to_copy {
            tensor[[0, 0, m, f]] = log_mel[[m, f]];
        }
    }

    tensor
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
