# Symptom to Disease Predictor

This project explores a symptom-to-disease dataset, trains a common-first disease classifier, and provides command-line and Streamlit-based tools for testing predictions.

## What is included

- `sympdis.csv` for the raw dataset.
- Cleaning and understanding scripts for inspecting the data.
- A GPU-capable training script for the main model.
- Tester scripts for saved model bundles.
- A Streamlit dashboard for interactive symptom input.

## Main scripts

- `cleaner_basic.py` prepares a cleaned version of the dataset.
- `common_first_predictor.py` trains the model and saves a checkpoint bundle.
- `test_common_model.py` tests the saved model bundle.
- `test_common_model_v2.py` tests the v2 model bundle.
- `streamlit_dashboard.py` launches the interactive UI.

## How to Import DATASET

1. Put `sympdis_clean_basic.csv` in the project root (same folder as `common_first_predictor.py`).
2. Keep the filename exactly as `sympdis_clean_basic.csv`.
3. Confirm the file exists before training.

Expected structure in the root folder:

```text
Symptom to Disease/
	common_first_predictor.py
	streamlit_dashboard.py
	test_common_model_v2.py
	sympdis_clean_basic.csv
```

Optional quick check:

```bash
python -c "import os; print(os.path.exists('sympdis_clean_basic.csv'))"
```

## Run the project

Install dependencies first:

```bash
pip install numpy pandas scikit-learn torch streamlit joblib
```

Train the model:

```bash
python common_first_predictor.py
```

Test the v2 model interactively:

```bash
python test_common_model_v2.py --interactive
```

Launch the Streamlit dashboard:

```bash
streamlit run streamlit_dashboard.py
```

The dashboard uses:

- Model path: `model_checkpoints/common_first_model_v2.pt` (or your selected model file)
- Cleaned data path: `sympdis_clean_basic.csv`

## Notes

- The model uses a priority/common-disease bias rather than raw class frequency alone.
- The dashboard supports top-k selection, symptom-profile blending, and prediction stability controls.
- Generated analysis artifacts are excluded from version control through `.gitignore`.

## Suggested GitHub setup

After verifying the repo locally, initialize git and push it to a new GitHub repository.
