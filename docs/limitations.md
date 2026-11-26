# Limitations

VoiceMarkers is a research and education demo. **It is not a medical device** and must not
be used to diagnose, screen for or rule out Parkinson's disease or any other condition.
These are the reasons its predictions on new recordings are not reliable, in rough order
of importance.

## 1. Very little data

- 195 recordings from **32 subject IDs**, only **8 of them healthy**. Every estimate has
  a wide margin, shown by the ± standard deviations in
  [`evaluation.md`](evaluation.md). The deployed model's specificity (healthy voices
  correctly recognised) varies by ±41 points between test folds.
- The dataset's description says 31 people (23 with Parkinson's), but the file encodes 32
  distinct subject IDs. Splits group on the IDs as they appear. If two IDs belong to
  one person, a little leakage remains, much less than in a row-level split.
- Little is known about the participants (age, sex, disease stage, medication), so the
  results can't be broken down by group or checked for bias.

## 2. The honest performance is modest

- Evaluated on **new people** (subject-wise), the best models reach about **60–65%
  balanced accuracy**, where 50% is chance. The paper's 87.18% accuracy was measured on
  a row-level split, i.e. on new recordings of people the model had already heard. See
  the [evaluation report](evaluation.md).
- No model is clearly better than the others: their differences are much smaller than
  the fold-to-fold spread.

## 3. Features differ from the training data

- **Different software:** the dataset's features were computed with MDVP (Kay Pentax).
  The app computes them with Praat. The definitions match, but the algorithms differ,
  so values are comparable in range, not identical.
- **No way to check one to one:** the UCI dataset contains features only, not audio, so
  we can't run Praat on the original recordings and compare. The
  [feature validation](../research/results/feature_validation.md) only shows that
  extracted values are *plausible*.
- **Fewer features:** 7 of the 22 features (NHR, RPDE, DFA, D2, PPE, spread1, spread2)
  can't be reproduced and aren't used by the deployed model.

## 4. Recording conditions

- The training recordings were made in a controlled setting. Phone and laptop
  microphones, browser audio processing, codecs (WebM and MP4 are lossy), background
  noise and the room all change jitter, shimmer and HNR.
- The app turns off the browser's echo cancellation, noise suppression and automatic
  gain where it can, but it can't undo the hardware.

## 5. Reading the output

- **The probability leans towards Parkinson's for most voices,** because 75% of the
  training recordings were from people with Parkinson's. The UI shows it next to the
  base rate and the model's training average, and the label comes from the model's own
  balanced decision, not from 50%.
- **Explanations describe the model, not the voice:** SHAP contributions say what this
  model does with the values. Correlated features (the jitter and shimmer variants) share
  credit between them.
- **Many things change a voice:** colds, fatigue, smoking, age and other conditions all
  affect these measures. Nothing here is specific to Parkinson's disease.

## 6. Engineering limits

- **Rate limiting** is in memory and per process (10 per minute per client IP).
- **Cold starts:** the free Hugging Face Space sleeps after about 48 hours idle, so the
  first request afterwards waits for it to start.
- **Microphone recording** is tested with a mocked `MediaRecorder`. Real-device behaviour
  varies by browser.
