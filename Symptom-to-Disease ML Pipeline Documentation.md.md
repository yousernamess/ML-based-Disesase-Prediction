# Master Documentation: Symptom-to-Disease Project

This is the single master document that combines:

- data understanding and cleaning findings,
- model-build decisions from analysis,
- and detailed technical explanations of the neural network pipeline.

## Part A: Data Understanding and Cleaning

## A1) Raw dataset findings

The raw dataset is a symptom-to-disease table with:

- Rows: 246,945
- Columns: 378
- Disease labels: 773
- Symptom columns: 377
- Missing values: 0

Key observations:

- Duplicate rows: 57,298
- Constant symptom columns: 49
- Invalid symptom rows (non 0/1): 0

Interpretation:

- The dataset is structurally usable (binary symptom format, no missing values),
- but contains substantial duplication and non-informative columns.

## A2) Cleaning decisions and exact changes

### Step 1: Remove exact duplicates

Why:

- Duplicate rows inflate repeated symptom patterns and can bias training.

Effect:

- 57,298 duplicate rows removed.

### Step 2: Validate binary symptom values

Why:

- Symptom matrix must remain binary for consistent modeling.

Effect:

- No rows removed (all symptom values already valid 0/1).

### Step 3: Remove constant symptom columns

Why:

- Constant features carry no predictive information.

Effect:

- 49 constant symptom columns removed.

### Step 4: Preserve disease labels

Why:

- Disease column is the target; it must stay unchanged.

Effect:

- Unique diseases stayed at 773.
- No disease labels were lost.

## A3) Cleaned dataset result

After cleaning:

- Rows: 189,647
- Columns: 329
- Disease labels: 773
- Symptom columns: 328
- Missing values: 0
- Duplicate rows: 0
- Constant symptom columns: 0

Retention:

- Rows kept: 76.80%
- Columns kept: 87.04%

## A4) What prevalence analysis showed

Top cleaned symptom prevalences:

- sharp abdominal pain: 13.26%
- vomiting: 11.61%
- cough: 9.97%
- headache: 9.90%
- nausea: 9.88%

Interpretation:

- Features are sparse and long-tailed.
- User-entered symptom sets are often partial/incomplete.

## Part B: Model Build Decisions Driven by Analysis

## B1) Why priority logic was needed

Disease class ranking showed that sample count does not always match desired practical priority.

Examples from class ranking:

- cystitis: 1219 (0.6428%)
- common cold: 813 (0.4287%)
- flu: 521 (0.2747%)
- gerd: 302 (0.1592%)
- diabetes: 1 (0.0005%)

Interpretation:

- Pure frequency-based modeling would over-favor classes that are common in the dataset, not necessarily the diseases you wanted to prioritize.

Decision:

- Use an explicit priority disease list (manual clinical/product priority), not only frequency thresholds.

## B2) Core model architecture

A multiclass PyTorch neural network:

- Input: binary symptom vector (all cleaned features)
- Hidden layers: 512 -> 256 -> 128
- Activations: ReLU
- BatchNorm: first 3 hidden blocks
- Dropout: 0.3 in first 2 hidden blocks
- Output: logits over all disease classes

Why this architecture:

- Handles high-dimensional sparse binary input.
- Can learn non-linear symptom interactions.
- Trains efficiently with GPU support.

## B3) Training-time class weighting

Base class weighting:

$$
w_c^{base} = \sqrt{\frac{N}{n_c}}
$$

Then normalized by mean weight.

Priority classes receive extra multiplier:

- common_weight_multiplier = 2.6

Final cap:

- max_class_weight = 10.0

Why:

- Base weighting helps long-tail imbalance.
- Priority multiplier enforces importance of selected diseases.
- Cap prevents unstable rare-class over-weighting.

Example for flu (521) vs cystitis (1219):

- Relative square-root term: $\sqrt{1219/521} \approx 1.53$
- With priority multiplier: $1.53 \times 2.6 \approx 3.98$ relative emphasis (before normalization/cap details).

## B4) Inference-time priority boost

After softmax, priority-class probabilities are multiplied by:

- inference_common_boost = 1.35

Then renormalized.

Why:

- Adds a controlled final ranking bias toward priority diseases.
- Complements training-time weighting.

## B5) Robustness choices for partial symptom input

### Symptom dropout augmentation

Settings:

- dropout_rate = 0.3
- dropout_copies = 1

Mechanism:

- In augmented copies, randomly drop some present symptoms (1 -> 0).

Why:

- Makes model robust when users provide incomplete symptom sets.

### Positive-only symptom profile blending

At inference:

- Build class symptom profiles from training data,
- blend model posterior with positive-only symptom match distribution.

Blend:

- positive_only_blend = 0.35

Why:

- Reduces abrupt rank collapse when one symptom is added.
- Rewards diseases whose historical profile better matches reported positives.

### Near-case logic

- If top-1 and top-2 are very close (near_margin = 0.05), flag both as high-priority candidates.

Why:

- Better decision support in ambiguous cases.

## B6) Evaluation and training setup

- Stratified split with test_size = 0.2
- Singleton classes (<2 samples) removed from evaluation prep
- Loss: weighted CrossEntropyLoss
- Optimizer: Adam
- LR scheduler: CosineAnnealingLR
- Learning rate: 1e-3
- Epochs: 40
- Batch size: 2048
- Device: CUDA if available, else CPU

Why singleton removal:

- Stratified train/test requires at least two samples per class.

## B7) Performance snapshot

From saved evaluation checkpoint:

- raw_top1_accuracy = 0.8269
- adjusted_top1_accuracy = 0.8261

Interpretation:

- Priority-aware adjustments preserve strong top-1 performance while improving practical ranking behavior for prioritized diseases.

## Part C: Technical Concepts (Plain English)

## C1) Multiclass classification

This is not yes/no classification.

The model predicts one probability per disease class, then ranks them.

## C2) Logits and softmax

Final network output is logits (raw scores), then softmax converts to probabilities:

$$
P(y=i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

## C3) Why ReLU

ReLU keeps positive signal and is efficient:

$$
f(x)=\max(0,x)
$$

## C4) Why BatchNorm

BatchNorm stabilizes internal activations, improving convergence speed and training stability.

## C5) Why model-layer dropout (0.3)

Dropout reduces overfitting by preventing over-reliance on a small set of hidden neurons.

## C6) Why class weighting works

Without weights, frequent classes dominate loss.

Weighted cross-entropy makes rare or priority classes matter more during optimization.

## C7) Why weight clipping is necessary

Very large rare-class weights can destabilize gradients.

Capping to 10.0 keeps learning stable while still helping tail classes.

## C8) Training augmentation vs model dropout

- Model dropout: inside network hidden layers.
- Symptom dropout augmentation: modifies training inputs.

They solve different problems and are both used.

## C9) Why positive-only blending helps

Pure softmax ranking can shift abruptly when one symptom changes.

Blending in positive-only profile match adds symptom-compatibility memory and stabilizes ranking.

## C10) Why confidence can look modest

Even good multiclass models can output moderate confidence because:

- many classes compete,
- symptoms overlap across diseases,
- long-tail classes increase ambiguity.

## C11) Why top-k is better than only top-1

In triage-like prediction tasks, several diseases can be close.

Top-3 plus near-case alerts is safer and more useful than strict single-label output.

## C12) End-to-end pipeline summary

1. Load cleaned dataset.
2. Remove singleton classes for valid stratified split.
3. Split train/test.
4. Build class weights (imbalance + priority multiplier + cap).
5. Apply symptom dropout augmentation.
6. Train NN with weighted cross-entropy + Adam + cosine LR schedule.
7. Save model state, encoder, features, class weights, and class profiles.
8. Inference: softmax -> priority boost -> profile blending -> top-k + near-case logic.

## Final Master Summary

The project combines:

- rigorous dataset cleanup,
- analysis-driven model design,
- explicit priority-aware weighting and boosting,
- robustness mechanisms for incomplete symptom input,
- and practical decision-oriented output behavior.

This is why the final system is stronger than a plain, frequency-only classifier trained on raw data.
