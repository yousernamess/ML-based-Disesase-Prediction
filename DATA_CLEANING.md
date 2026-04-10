# Data Understanding and Cleaning

This document explains what was learned from the dataset, what those findings meant, and how the cleaning steps were chosen from those findings.

## 1. What we found about the raw dataset

The raw dataset is a large symptom-to-disease table.

### Main numbers

- Rows: 246,945
- Columns: 378
- Disease labels: 773 unique diseases
- Symptom columns: 377
- Missing values: 0

### What the dataset looked like

- The first column is the disease label.
- The remaining columns are symptom indicators.
- Symptom values are supposed to be binary, with `0` meaning absent and `1` meaning present.

### What stood out immediately

- The dataset had a lot of duplicate rows: 57,298 exact duplicates.
- There were 49 symptom columns that never changed at all.
- There were no missing values.
- There were no invalid symptom values outside `0` and `1`.

## 2. What that understanding meant

The raw dataset was already structured well, but it had obvious noise.

### Meaning of the duplicate rows

The duplicate rows meant the same symptom pattern appeared more than once.

What that tells us:

- Some diseases or symptom combinations were repeated in the raw data.
- If left in place, duplicates would distort the training signal and overstate repeated patterns.

### Meaning of the constant symptom columns

The 49 constant symptom columns were not useful for learning.

What that tells us:

- Those columns carried no variation.
- A column that is always `0` or always `1` cannot help the model separate diseases.
- Keeping them would add size without adding information.

### Meaning of the binary validation result

The symptom values were already clean in the sense that they matched the expected `0/1` format.

What that tells us:

- There was no need to invent values, impute values, or recode the symptom matrix.
- The dataset was already consistent in its symptom encoding.

### Meaning of the disease label distribution

The dataset contained 773 diseases, but the sample counts were uneven.

What that tells us:

- The dataset is long-tailed.
- Some diseases appear many times while others appear much less often.
- This matters later for model training because class imbalance can affect which diseases the model prefers.

### Meaning of the symptom prevalence ranking

The most common symptoms in the cleaned data were:

- sharp abdominal pain: 13.26%
- vomiting: 11.61%
- cough: 9.97%
- headache: 9.90%
- nausea: 9.88%
- back pain: 9.80%
- sharp chest pain: 9.68%
- fever: 8.81%
- shortness of breath: 8.59%
- nasal congestion: 6.92%

What that tells us:

- The dataset has a symptom core that appears a lot across diseases.
- Many other symptoms are much rarer.
- The data is not evenly distributed across features.

## 3. What we decided to clean

From those findings, the cleaning plan was simple and conservative.

### Cleaning step 1: remove exact duplicates

Why:

- Duplicate rows would exaggerate repeated symptom patterns.
- They would make the dataset look larger than it really is.

What changed:

- 57,298 duplicate rows were removed.

### Cleaning step 2: keep only valid binary symptom rows

Why:

- The model expects symptom features to be binary.
- Any non-binary value would be a data quality problem.

What changed:

- No rows were removed for invalid symptom values because none were found.

### Cleaning step 3: remove constant symptom columns

Why:

- Constant columns do not help prediction.
- They increase dimensionality without adding information.

What changed:

- 49 constant symptom columns were removed.

### Cleaning step 4: preserve the disease label exactly

Why:

- The disease column is the target label, not a feature.
- It must remain unchanged for training and evaluation.

What changed:

- The disease labels were preserved as-is.
- No disease label was dropped.

## 4. What the cleaned dataset looks like

After cleaning, the dataset became:

- Rows: 189,647
- Columns: 329
- Unique diseases: 773
- Symptom columns: 328
- Missing values: 0
- Duplicate rows: 0
- Constant symptom columns: 0

### How much changed

- Rows kept: 76.80% of the original data
- Columns kept: 87.04% of the original data

### What this means

- The cleaning removed only obvious noise.
- It did not change the disease coverage.
- It kept the symptom matrix valid and smaller.

## 5. Final understanding after cleaning

After cleaning, the dataset was better suited for modeling because:

- duplicates were gone,
- invalid symptom values were not present,
- useless constant symptom columns were removed,
- the disease labels stayed intact,
- and the binary symptom structure was preserved.

The cleaned data still had the same overall disease diversity, but it was less noisy and more efficient for training and analysis.

## 6. Why this cleaning approach made sense

The cleaning was intentionally basic.

It focused on removing only what was clearly unnecessary or harmful:

- repeated rows,
- non-informative columns,
- and any invalid feature values if they had existed.

It did not try to reshape the meaning of the dataset.

That was the right choice because the data already had a structured binary format, so the main task was to clean it, not redesign it.

## 7. Short summary

The raw dataset had 246,945 rows, 378 columns, 773 diseases, no missing values, 57,298 duplicates, and 49 constant symptom columns. Cleaning removed the duplicates and constant columns, kept the disease labels unchanged, and left a 189,647-row cleaned dataset with 328 symptom features ready for modeling.