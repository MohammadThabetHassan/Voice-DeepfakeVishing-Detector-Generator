# Enhancements Review And Threshold Tuning (March 3, 2026)

## Scope

This review was run after retraining on a broader external dataset and validating on user-provided challenge files.

## Key Findings

1. Main root cause of earlier false positives was domain mismatch.
   - Old models were trained on narrow data and overfit dataset artifacts.
   - Real-world human audio (YouTube style, compression, mastering, room acoustics) was out-of-domain.

2. Preprocessing consistency mattered.
   - Keeping valid WAV uploads as-is (no forced re-encode) improved reliability.
   - Multi-window voiced inference reduced single-segment bias.

3. Current probability threshold trade-off is real.
   - Raising threshold lowers false positives but increases missed fakes.

4. There is still technical debt to address.
   - `enhanced` feature dimensionality is inconsistent across training/inference docs and code paths.
   - Model artifact/version tracking is manual.

## Priority Enhancements

1. Data and evaluation quality
   - Maintain a speaker-disjoint train/validation/test split.
   - Keep a permanent out-of-domain benchmark set (YouTube/podcast/phone-call style).
   - Track per-source metrics (not only global metrics).

2. Decision policy and calibration
   - Keep configurable threshold profiles: `balanced`, `low_fp`, `high_recall`.
   - Add an `uncertain` state near threshold to avoid overconfident binary output.
   - Calibrate probabilities (Platt/isotonic) using held-out validation data.

3. Feature/schema consistency
   - Unify `enhanced` feature dimensionality between `training/train.py`, `backend/app.py`, and docs.
   - Add strict schema checks when loading models (fail fast on mismatch).

4. MLOps hygiene
   - Introduce model manifest/versioning (`model_id`, training data hash, training date, threshold).
   - Automate regression checks before promoting a model to `deepfake_detector_best.pkl`.

5. Product-level feedback
   - Return `fake_probability` + threshold + window stats in UI.
   - Show warning when score is borderline or likely out-of-domain.

## Threshold Tuning Run

Artifacts generated:
- `models/threshold_tuning_20260303.json`
- `models/threshold_tuning_samples_20260303.csv`

Calibration dataset:
- `training/data_external` (`1076` real, `993` fake)
- plus user challenge files where available

Observed trade-off examples:
- Threshold `0.70`: FPR `0.0056`, fake recall `0.6281`
- Threshold `0.80`: FPR `0.0046`, fake recall `0.5638`
- Threshold `0.85`: FPR `0.0028`, fake recall `0.5106`

Operational choice for this repo update:
- Set default `DETECTION_FAKE_THRESHOLD=0.8` in backend for lower false positives.
- Keep env override for local tuning:
  - `DETECTION_FAKE_THRESHOLD=0.7` for higher fake recall
  - `DETECTION_FAKE_THRESHOLD=0.85` for aggressive low-FP behavior

## Post-Update Validation

With the deployed retrained hybrid model and default threshold `0.8`:
- `OSR_us_000_0010_8k.wav` -> real
- `OSR_us_000_0030_8k.wav` -> real
- `generated_voice(1).wav` -> fake
- `YTDown...online-video-cutter...(1).wav` -> real

## Additional Engineering Updates (March 3, 2026)

Implemented after this review:

1. Web data collection pipeline
   - Added `scripts/collect_web_audio.py` to collect labeled real/fake audio from:
     - YouTube search queries and direct URLs (via `yt-dlp`)
     - Podcast RSS feeds (episode enclosure downloads)
   - Normalizes to mono 16 kHz WAV and exports `training/web_data/metadata.csv`.

2. Feature-schema consistency fix
   - Fixed backend enhanced inference schema to match training (`75` dims).
   - Kept backward compatibility for legacy `30`-dim enhanced artifacts.

3. Uncertainty-aware detection output
   - `/detect` now returns:
     - `prediction`: `real` / `fake` / `uncertain`
     - `base_prediction`, `is_uncertain`, `fake_probability`, `threshold`, `uncertain_margin`

4. Model manifest/versioning
   - Training now stores metadata in each model artifact:
     - `model_id`, `training_date_utc`, `training_data_hash`, `recommended_threshold_profiles`
   - Added `models/model_manifest.json` as a consolidated run manifest.

5. Backend schema validation
   - Added strict model schema checks at load time (`DETECTION_STRICT_SCHEMA_CHECK=true` by default).
   - Prevents loading artifacts with incompatible feature dimensions/order.
