use rage_core::RageError;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decode an audio file into interleaved f32 samples.
///
/// Returns (samples, sample_rate, num_channels).
pub fn decode_file(path: &Path) -> Result<(Vec<f32>, u32, u16), RageError> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| RageError::Decode(format!("failed to probe format: {e}")))?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| RageError::Decode("no audio tracks found".into()))?;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| RageError::Decode("unknown sample rate".into()))?;

    let channels = track
        .codec_params
        .channels
        .map(|ch| ch.count() as u16)
        .unwrap_or(1);

    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| RageError::Decode(format!("failed to create decoder: {e}")))?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(RageError::Decode(format!("packet read error: {e}"))),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(symphonia::core::errors::Error::DecodeError(e)) => {
                tracing::warn!("decode error (skipping packet): {e}");
                continue;
            }
            Err(e) => return Err(RageError::Decode(format!("decode error: {e}"))),
        };

        let spec = *decoded.spec();
        let n_frames = decoded.frames();

        if n_frames == 0 {
            continue;
        }

        let mut sample_buf = SampleBuffer::<f32>::new(n_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(sample_buf.samples());
    }

    if all_samples.is_empty() {
        return Err(RageError::AudioTooShort {
            min_samples: 1,
            actual_samples: 0,
        });
    }

    Ok((all_samples, sample_rate, channels))
}
