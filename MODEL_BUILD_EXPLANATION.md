# Neural Network Model Build: What We Learned and Why We Chose Each Step

## Goal

The objective was not just to train a high-accuracy classifier, but to make a symptom-to-disease model that:

- works on a large multi-class dataset,
- gives top-k outputs,
- remains useful when user symptom input is incomplete,
- and does not ignore medically important diseases just because their sample count is low.

## What the analysis tables told us

### 1) Class ranking showed that sample count does not match medical commonness

From the disease class ranking table on cleaned data:

- `cystitis`: 1219 samples (0.6428%)
- `acute bronchitis`: 1213 samples (0.6396%)
- `pneumonia`: 1212 samples (0.6391%)
- `common cold`: 813 samples (0.4287%)
- `flu`: 521 samples (0.2747%)
- `gastroesophageal reflux disease (gerd)`: 302 samples (0.1592%)
- `diabetes`: 1 sample (0.0005%)

Interpretation:

- The dataset is synthetic/imbalanced in a way where some clinically common diseases are not top-frequency classes.
- If we used pure frequency logic, the model would over-favor classes that are common in this dataset, not common in real-world usage expectations.

Decision taken:

- We used an explicit priority disease list instead of letting frequency define "common".
- This keeps the model behavior aligned with product intent, not only dataset counts.

### 2) Feature prevalence table showed a sparse symptom space

From feature prevalence ranking:

- Top symptoms are relatively low prevalence (for example `sharp abdominal pain` at 13.26%, `vomiting` at 11.61%, `cough` at 9.97%).
- Many symptoms are very rare (long tail).

Interpretation:

- Input vectors are sparse and often incomplete in real use.
- A model trained on full symptom vectors can be brittle when the user adds/removes only one symptom.

Decisions taken:

- Added symptom dropout augmentation during training.
- Added positive-only symptom profile blending at inference.
- Added near-case logic for close top-1/top-2 outcomes.

## Model choices and exact settings

## 1) Model type and architecture

We built a multiclass neural network in PyTorch:

- Input: all cleaned symptom features (binary columns)
- Hidden layers: 512 -> 256 -> 128
- Activations: ReLU
- Stabilization: BatchNorm after first three linear layers
- Regularization: Dropout 0.3 on first two hidden blocks
- Output: logits over all disease classes

Why:

- Handles high-dimensional sparse binary inputs.
- Fast on GPU.
- Enough capacity for nonlinear interactions between symptoms while still trainable at this dataset scale.

## 2) Priority-aware class weighting (training time)

Base class weight formula:

- $w_c^{base} = \sqrt{N / n_c}$
- Then normalized by mean weight across classes.

Priority boost:

- If class is in the priority list, multiply by `common_weight_multiplier = 2.6`.

Cap:

- Final class weight is clipped to `max_class_weight = 10.0`.

Why:

- Base weighting helps with long-tail imbalance.
- Priority multiplier explicitly pushes medically important classes upward.
- Cap prevents unstable gradients from extreme rare-class weights.

Concrete example from your table:

- `flu` has 521 samples while classes like `cystitis` have 1219.
- Count-based ratio inside the square-root term is roughly $\sqrt{1219/521} \approx 1.53$ in favor of `flu`.
- With priority multiplier, this becomes roughly $1.53 \times 2.6 \approx 3.98$ relative emphasis vs a non-priority class with 1219 samples (before final normalization/clipping details).

This is exactly why the model does not let `flu` disappear simply because it has fewer rows than many non-priority classes.

## 3) Priority-aware bias (inference time)

After softmax probabilities are produced, classes in the priority list are multiplied by:

- `inference_common_boost = 1.35`

Then probabilities are renormalized.

Why:

- Training-time weighting shapes the model globally.
- Inference-time boost gives a controlled, transparent last-mile bias toward priority diseases.

## 4) Symptom dropout augmentation

Training augmentation settings:

- `dropout_rate = 0.3`
- `dropout_copies = 1`

Mechanism:

- For augmented copies, randomly switch some present symptoms (`1`) to absent (`0`).

Why:

- Users often provide incomplete symptom sets.
- This teaches the model to remain stable when symptoms are missing or added incrementally.

## 5) Positive-only symptom profile blending

At inference, we build class symptom profiles from training data and combine:

- priority-biased model posterior,
- positive-only symptom match distribution.

Blend setting:

- `positive_only_blend = 0.35`

Why:

- Helps prevent abrupt rank collapse when one symptom is added.
- Rewards diseases whose historical symptom profile matches the selected positive symptoms.

## 6) Evaluation split handling

Before train/test split:

- Singleton classes (sample count < 2) are removed from evaluation dataset.

Why:

- Stratified split requires at least two samples per class.
- This keeps evaluation statistically valid.

## 7) Optimization and training schedule

- Loss: CrossEntropyLoss with class weights
- Optimizer: Adam
- Learning rate: `1e-3`
- Scheduler: CosineAnnealingLR
- Epochs: `40`
- Batch size: `2048`
- Device: CUDA if available, else CPU

Why:

- Stable and efficient training on large tabular multiclass data.
- 40 epochs was selected after experiments where pushing higher epochs hurt generalization.

## 8) Output behavior and UX controls

Model behavior is designed around practical triage use:

- top-k output (`top_k = 3` default),
- near-case margin (`near_margin = 0.05`),
- optional priority boosting and blend controls in testing/dashboard tools.

Why:

- In medical-like tasks, close alternatives matter.
- Top-3 with margin warning is safer than only top-1.

## What this build achieved

The model pipeline now combines:

- data-driven learning,
- imbalance correction,
- explicit priority policy,
- robustness to partial symptom input,
- and interpretable post-processing.

In short: analysis of class/sample imbalance and sparse symptom prevalence directly drove every major NN design choice (weighting, boosting, augmentation, and blending), including the explicit `2.6` training multiplier and `1.35` inference multiplier for priority diseases.
