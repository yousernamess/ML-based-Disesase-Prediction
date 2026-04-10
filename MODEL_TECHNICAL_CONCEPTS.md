# Model Technical Concepts: Plain-English Deep Explanation

This document explains the technical parts of your model in simple language, step by step.

## 1) Big Picture: What the model is doing

You give the model a list of symptoms.

The model converts that list into a long binary vector:

- `1` means symptom is present
- `0` means symptom is absent

Then it predicts probabilities for all diseases and returns the top ones.

You can think of it like this:

- Input: symptom checklist
- Brain: neural network + weighting logic
- Output: ranked disease probabilities

## 2) Why this is a multiclass neural network

You have many disease labels (hundreds of classes), not just yes/no.

So this is a multiclass classification problem.

Your network outputs one score per disease. Those scores are converted to probabilities.

## 3) Architecture explained (512 -> 256 -> 128)

The architecture is:

- Input layer: all symptom features
- Hidden layer 1: 512 units
- Hidden layer 2: 256 units
- Hidden layer 3: 128 units
- Output layer: number of diseases

What hidden layers do:

- They learn combinations of symptoms that are useful for prediction.
- Early layer learns simple patterns.
- Later layers learn more disease-specific interactions.

Why this size is reasonable:

- Large enough to model non-linear symptom interactions.
- Small enough to train efficiently.

## 4) ReLU activation

ReLU is the activation function:

- Formula: $f(x) = \max(0, x)$

Why used:

- Fast to compute.
- Helps deep nets train better than older activations in many cases.
- Keeps useful positive signal and suppresses negative noise.

## 5) BatchNorm: why normalization is inside the network

BatchNorm normalizes intermediate layer outputs during training.

What this helps with:

- More stable gradients.
- Faster convergence.
- Less sensitivity to bad scale in internal activations.

Practical effect:

- Training becomes smoother and less likely to oscillate.

## 6) Dropout in the network (0.3)

Dropout randomly drops some hidden units during training.

- Dropout 0.3 means around 30% are randomly ignored on each pass.

Why this helps:

- Reduces overfitting.
- Prevents the model from depending too much on a small set of neurons.

Important distinction:

- This is model-layer dropout, not symptom dropout augmentation.
- Your pipeline uses both types for different reasons.

## 7) Output logits, softmax, and probabilities

The final layer gives logits (raw scores).

Softmax turns logits into probabilities that sum to 1:

$$
P(y=i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Where:

- $z_i$ is logit for class $i$
- $P(y=i)$ is model probability for class $i$

Then classes are ranked by probability.

## 8) Class imbalance and why class weights are needed

Your dataset has long-tail class counts.

If no class weighting is used:

- Frequent classes dominate loss.
- Rare classes are under-learned.

So you weighted classes inversely by class frequency (with square root smoothing):

$$
w_c^{base} = \sqrt{\frac{N}{n_c}}
$$

Where:

- $N$ = total training samples
- $n_c$ = samples for class $c$

Then weights are normalized by mean.

Why square-root and not direct inverse:

- Full inverse can be too extreme.
- Square root gives a softer, more stable correction.

## 9) Priority-aware class weighting (multiplier 2.6)

You manually defined priority diseases.

For those classes only, training weight is multiplied by 2.6.

Why this exists:

- In your synthetic dataset, medically common/important diseases are not always frequent.
- You wanted model behavior to match clinical priority, not raw dataset frequency.

So your final training emphasis is:

- Base imbalance correction
- Plus extra priority multiplier
- Then clipped to max 10.0

## 10) Why clip class weights (max 10.0)

If a class is extremely rare, its computed weight can become huge.

Huge weights can cause:

- unstable gradients
- noisy updates
- over-correction

Clipping to 10.0 keeps training stable while still helping rare classes.

## 11) Training-time symptom dropout augmentation

This is a separate step from neural-network dropout.

You create augmented copies of training rows, then randomly turn some present symptoms (1) into absent (0).

Settings:

- `dropout_rate = 0.3`
- `dropout_copies = 1`

Meaning:

- One additional augmented copy
- About 30% of positive symptoms are removed in that copy

Why this helps:

- Real users provide incomplete symptom lists.
- Model becomes robust to missing symptoms.
- Predictions become less brittle when users add/remove a symptom.

## 12) Why singleton classes are removed before stratified split

A singleton class has only 1 row.

Stratified train/test split tries to preserve class distribution in both splits.

With one sample:

- You cannot place that class in both train and test.
- Stratification breaks or becomes invalid.

So singleton classes are removed from evaluation split prep.

Important:

- This is mostly an evaluation practicality step.
- It avoids unstable or impossible split behavior.

## 13) Loss function: weighted CrossEntropy

CrossEntropy compares predicted class distribution against the true class.

Weighted CrossEntropy multiplies each sample’s loss by class weight.

Why this is ideal here:

- Multiclass task.
- Imbalanced labels.
- Priority-aware weighting can be injected cleanly.

## 14) Optimizer: Adam

Adam adapts learning rates per parameter using moving averages of gradients.

Why Adam was a good choice:

- Works well out-of-the-box for many deep nets.
- Stable on sparse-ish tabular inputs.
- Faster convergence than plain SGD in many cases.

## 15) Learning-rate scheduler: CosineAnnealingLR

Learning rate starts higher and gradually decays with a cosine schedule.

Why this helps:

- Large steps early for fast learning.
- Smaller steps later for fine-tuning.
- Can improve final convergence stability.

## 16) Epochs, batch size, and device

Current defaults:

- Epochs: 40
- Batch size: 2048
- Device: CUDA if available, else CPU

Why these matter:

- Epochs control how many full passes over data.
- Batch size controls gradient estimate smoothness and throughput.
- GPU massively speeds training for this workload.

Why 40 epochs:

- You observed better generalization than an overlong run.

## 17) Priority-aware inference boost (1.35)

After softmax, you multiply priority-class probabilities by 1.35 and renormalize.

This is post-processing, not retraining.

Why useful:

- Gives a controlled preference to priority classes at decision time.
- Complements training-time weights.

Difference from training weight:

- Training weight changes learning process.
- Inference boost changes ranking at prediction time.

## 18) Positive-only symptom profile blending (0.35)

You build class symptom profiles from training data.

For each disease, a profile is mean symptom presence probability per feature.

At inference, you compute:

- model posterior (after priority boost)
- positive-only match distribution (how well reported positive symptoms match each class profile)

Then blend:

$$
P_{final} = (1-\alpha)P_{model} + \alpha P_{pos}
$$

Where $\alpha = 0.35$.

Why this helps:

- Reduces abrupt rank collapse when one symptom is added.
- Uses symptom compatibility signal directly.

## 19) Top-k and near-case logic

Instead of only top-1, you return top-k (default 3).

Near-case check:

- If top-1 and top-2 are very close (margin threshold), flag that both are important.

Why this is safer:

- Medical-like predictions often have close alternatives.
- A strict single-label answer can hide uncertainty.

## 20) Why confidence can still look low

Even with good accuracy, probabilities can be moderate because:

- Many classes compete in softmax.
- Symptoms overlap heavily across diseases.
- Long-tail classes make separation harder.

So lower confidence does not always mean wrong model.

## 21) End-to-end flow summary

1. Read cleaned data.
2. Remove singleton classes for stable split.
3. Stratified train/test split.
4. Build class weights from imbalance + priority multiplier.
5. Augment training data with symptom dropout.
6. Train NN with weighted CrossEntropy + Adam + cosine scheduler.
7. Save model, label encoder, feature list, weights, and class profiles.
8. At inference: model probability -> priority boost -> profile blending -> top-k + near-case note.

## 22) Practical intuition

Your model is not just "a neural net".

It is a layered decision system:

- Neural net learns symptom-to-disease mapping.
- Class weights fix imbalance and priority alignment.
- Inference boost pushes priority diseases when appropriate.
- Profile blending stabilizes symptom-based ranking behavior.
- Top-k + near-case logic improves practical usability.

That combination is why the system is much more useful than a plain raw classifier.
