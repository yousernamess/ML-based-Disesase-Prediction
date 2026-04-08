# import argparse
# import json
# import os
# import shutil
# from datetime import datetime
# from typing import Dict, List, Tuple

# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
# from sklearn.model_selection import train_test_split


# PRIORITY_DISEASES = {
#     "common cold",
#     "flu",
#     "pneumonia",
#     "acute bronchitis",
#     "asthma",
#     "chronic obstructive pulmonary disease (copd)",
#     "seasonal allergies (hay fever)",
#     "anxiety",
#     "depression",
#     "diabetes",
#     "infectious gastroenteritis",
#     "chronic constipation",
#     "gastritis",
#     "gastroesophageal reflux disease (gerd)",
#     "urinary tract infection",
#     "eczema",
#     "psoriasis",
#     "acne",
#     "fungal infection of the skin",
#     "conjunctivitis",
#     "dental caries",
#     "tooth abscess",
#     "gum disease",
#     "ear infection (otitis media)",
#     "tonsillitis",
#     "strep throat",
#     "migraine",
#     "tension headache",
#     "chronic back pain",
#     "osteoarthritis",
#     "kidney stone",
#     "hemorrhoids",
# }

# PRIORITY_ALIASES = {
#     "ear infection (otitis media)": "otitis media",
# }


# def build_class_weights(
#     y_train: pd.Series,
#     common_classes: set,
#     common_weight_multiplier: float,
#     max_class_weight: float,
# ) -> Dict[str, float]:
#     counts = y_train.value_counts()
#     base = np.sqrt(len(y_train) / counts)
#     base = base / base.mean()
#     weights = base.to_dict()
#     for cls in common_classes:
#         if cls in weights:
#             weights[cls] *= common_weight_multiplier
#     for cls in list(weights.keys()):
#         weights[cls] = float(min(weights[cls], max_class_weight))
#     return weights


# def augment_with_symptom_dropout(
#     x_train: pd.DataFrame,
#     y_train: pd.Series,
#     seed: int,
#     dropout_rate: float,
#     copies: int,
# ) -> Tuple[pd.DataFrame, pd.Series]:
#     if copies <= 0 or dropout_rate <= 0:
#         return x_train, y_train
#     rng = np.random.default_rng(seed)
#     augmented_x_parts = [x_train]
#     augmented_y_parts = [y_train]
#     values = x_train.to_numpy(copy=True)
#     for _ in range(copies):
#         dropped = values.copy()
#         ones_mask = dropped == 1
#         random_mask = rng.random(dropped.shape) < dropout_rate
#         dropped[np.logical_and(ones_mask, random_mask)] = 0
#         augmented_x_parts.append(pd.DataFrame(dropped, columns=x_train.columns))
#         augmented_y_parts.append(y_train.reset_index(drop=True))
#     x_aug = pd.concat(augmented_x_parts, axis=0, ignore_index=True)
#     y_aug = pd.concat(augmented_y_parts, axis=0, ignore_index=True)
#     return x_aug, y_aug


# def find_common_classes(y: pd.Series, min_samples: int) -> set:
#     counts = y.value_counts()
#     return set(counts[counts >= min_samples].index)


# def load_priority_classes(priority_file: str) -> set:
#     if not priority_file:
#         return set(PRIORITY_DISEASES)
#     classes = set()
#     with open(priority_file, "r", encoding="utf-8") as f:
#         for line in f:
#             name = line.strip()
#             if name:
#                 classes.add(name)
#     return classes


# def resolve_priority_classes_in_dataset(all_labels: pd.Series, requested: set) -> Tuple[set, List[str]]:
#     label_set = set(all_labels.unique())
#     found = set()
#     missing = []
#     for d in requested:
#         if d in label_set:
#             found.add(d)
#             continue
#         alias = PRIORITY_ALIASES.get(d)
#         if alias and alias in label_set:
#             found.add(alias)
#             continue
#         missing.append(d)
#     missing = sorted(missing)
#     return found, missing


# def apply_common_bias(
#     probabilities: np.ndarray,
#     classes: np.ndarray,
#     common_classes: set,
#     inference_common_boost: float,
# ) -> np.ndarray:
#     class_multiplier = np.ones(len(classes), dtype=float)
#     for idx, cls in enumerate(classes):
#         if cls in common_classes:
#             class_multiplier[idx] = inference_common_boost
#     adjusted = probabilities * class_multiplier
#     row_sum = adjusted.sum(axis=1, keepdims=True)
#     adjusted = adjusted / np.where(row_sum == 0, 1.0, row_sum)
#     return adjusted


# def build_class_symptom_profiles(
#     x_train: pd.DataFrame,
#     y_train: pd.Series,
#     feature_cols: List[str],
# ) -> Dict[str, np.ndarray]:
#     train_with_target = x_train.copy()
#     train_with_target["__target__"] = y_train.values
#     grouped = train_with_target.groupby("__target__")[feature_cols].mean()
#     return {cls: grouped.loc[cls].to_numpy(dtype=float) for cls in grouped.index}


# def positive_only_match_distribution(
#     reported_indices: np.ndarray,
#     classes: np.ndarray,
#     class_profiles: Dict[str, np.ndarray],
#     eps: float = 1e-6,
# ) -> np.ndarray:
#     if reported_indices.size == 0:
#         return np.ones(len(classes), dtype=float) / len(classes)
#     scores = np.zeros(len(classes), dtype=float)
#     for i, cls in enumerate(classes):
#         profile = class_profiles.get(str(cls))
#         if profile is None:
#             scores[i] = eps
#             continue
#         probs = np.clip(profile[reported_indices], eps, 1.0)
#         scores[i] = float(np.exp(np.mean(np.log(probs))))
#     total = scores.sum()
#     if total <= 0:
#         return np.ones(len(classes), dtype=float) / len(classes)
#     return scores / total


# def print_metrics(
#     y_true: pd.Series,
#     prob_raw: np.ndarray,
#     prob_adjusted: np.ndarray,
#     classes: np.ndarray,
# ) -> None:
#     pred_raw = classes[np.argmax(prob_raw, axis=1)]
#     pred_adjusted = classes[np.argmax(prob_adjusted, axis=1)]
#     top3_raw = top_k_accuracy_score(y_true, prob_raw, k=3, labels=classes)
#     top3_adjusted = top_k_accuracy_score(y_true, prob_adjusted, k=3, labels=classes)
#     print("\nEvaluation Metrics")
#     print("-" * 80)
#     print(f"Raw top-1 accuracy: {accuracy_score(y_true, pred_raw):.4f}")
#     print(f"Raw top-3 accuracy: {top3_raw:.4f}")
#     print(f"Raw macro F1: {f1_score(y_true, pred_raw, average='macro', zero_division=0):.4f}")
#     print(f"\nCommon-aware top-1 accuracy: {accuracy_score(y_true, pred_adjusted):.4f}")
#     print(f"Common-aware top-3 accuracy: {top3_adjusted:.4f}")
#     print(f"Common-aware macro F1: {f1_score(y_true, pred_adjusted, average='macro', zero_division=0):.4f}")


# def top_k_predictions(
#     prob_row: np.ndarray,
#     classes: np.ndarray,
#     k: int,
# ) -> List[Tuple[str, float]]:
#     top_idx = np.argsort(prob_row)[::-1][:k]
#     return [(str(classes[i]), float(prob_row[i])) for i in top_idx]


# def parse_symptom_input(symptom_text: str) -> List[str]:
#     return [s.strip().lower() for s in symptom_text.split(",") if s.strip()]


# def build_feature_vector(feature_cols: List[str], symptoms: List[str]) -> pd.DataFrame:
#     feature_set = set(symptoms)
#     row = {col: 1 if col.lower() in feature_set else 0 for col in feature_cols}
#     return pd.DataFrame([row])


# def symptom_indices(feature_cols: List[str], symptoms: List[str]) -> np.ndarray:
#     feature_map = {c.lower(): i for i, c in enumerate(feature_cols)}
#     idx = [feature_map[s] for s in symptoms if s in feature_map]
#     return np.array(idx, dtype=int)


# def show_prediction_output(preds: List[Tuple[str, float]], near_margin: float) -> None:
#     print("\nTop predictions (common-aware)")
#     print("-" * 80)
#     for rank, (label, conf) in enumerate(preds, start=1):
#         print(f"{rank}. {label} | confidence={conf:.4f}")
#     if len(preds) >= 2 and (preds[0][1] - preds[1][1]) <= near_margin:
#         print(
#             f"\nNear spot-on case: top-2 are close (margin <= {near_margin:.3f}). "
#             "Use both as high-priority candidates."
#         )


# def log_step(step: str, detail: str = "") -> None:
#     now = datetime.now().strftime("%H:%M:%S")
#     if detail:
#         print(f"[{now}] {step} | {detail}", flush=True)
#     else:
#         print(f"[{now}] {step}", flush=True)


# def save_checkpoint(checkpoint_dir: str, name: str, payload: Dict) -> None:
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     path = os.path.join(checkpoint_dir, f"{name}.json")
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, indent=2)
#     log_step("Checkpoint saved", path)


# def gpu_runtime_note() -> None:
#     has_nvidia = shutil.which("nvidia-smi") is not None
#     if has_nvidia:
#         log_step("GPU detected", "NVIDIA device found, but this script uses scikit-learn SGDClassifier (CPU path).")
#     else:
#         log_step("GPU note", "No NVIDIA runtime detected in PATH; running CPU path.")


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Common-first disease predictor with top-3 confidence output.")
#     parser.add_argument("--input", default="sympdis_clean_basic.csv", help="Input CSV path")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
#     parser.add_argument("--common-min-samples", type=int, default=500, help="Class count threshold (used only when --common-source=frequency)")
#     parser.add_argument("--common-source", choices=["priority", "frequency"], default="priority", help="How to choose common diseases for bias/weighting")
#     parser.add_argument("--priority-file", default="", help="Optional text file with one priority disease per line")
#     parser.add_argument("--common-weight-multiplier", type=float, default=1.8, help="Training weight multiplier for common diseases")
#     parser.add_argument("--inference-common-boost", type=float, default=1.15, help="Prediction-time probability boost for common diseases")
#     parser.add_argument("--max-class-weight", type=float, default=10.0, help="Cap for class weights")
#     parser.add_argument("--top-k", type=int, default=3, help="How many predictions to return")
#     parser.add_argument("--near-margin", type=float, default=0.05, help="Top-1 vs top-2 margin for near spot-on note")
#     parser.add_argument("--dropout-rate", type=float, default=0.3, help="Fraction of present symptoms to randomly hide during augmentation")
#     parser.add_argument("--dropout-copies", type=int, default=1, help="How many dropout-augmented copies of training data to add")
#     parser.add_argument("--positive-only-blend", type=float, default=0.35, help="Blend weight for positive-only matching at inference")
#     parser.add_argument("--symptoms", default="", help="Comma-separated symptom names for one custom prediction")
#     parser.add_argument("--row-index", type=int, default=-1, help="Optional dataset row index to predict")
#     parser.add_argument("--checkpoint-dir", default="model_checkpoints", help="Directory to save checkpoint files")
#     parser.add_argument("--model-path", default="model_checkpoints/common_first_model.joblib", help="Path to save trained model")
#     args = parser.parse_args()

#     log_step("Run started")
#     gpu_runtime_note()

#     log_step("Loading dataset", args.input)
#     df = pd.read_csv(args.input)
#     if df.shape[1] < 2:
#         raise ValueError("Dataset must have target + features.")

#     target_col = df.columns[0]
#     feature_cols = list(df.columns[1:])

#     log_step("Preparing evaluation dataset", "Removing singleton classes")
#     counts = df[target_col].value_counts()
#     valid_classes = counts[counts >= 2].index
#     eval_df = df[df[target_col].isin(valid_classes)].copy()

#     x = eval_df[feature_cols]
#     y = eval_df[target_col]

#     log_step("Splitting train/test", f"test_size={args.test_size}")
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=args.seed, stratify=y)

#     if args.common_source == "priority":
#         requested_priority = load_priority_classes(args.priority_file)
#         common_classes, missing_priority = resolve_priority_classes_in_dataset(y_train, requested_priority)
#         log_step("Common source", "priority list")
#         log_step("Priority diseases found in training labels", str(len(common_classes)))
#         if missing_priority:
#             log_step("Priority diseases not found", str(len(missing_priority)))
#             print("Missing priority disease labels:")
#             for name in missing_priority:
#                 print(f"- {name}")
#     else:
#         common_classes = find_common_classes(y_train, args.common_min_samples)
#         log_step("Common source", f"frequency threshold >= {args.common_min_samples}")

#     if not common_classes:
#         raise ValueError("No common/priority diseases resolved. Check labels or priority file.")

#     class_weights = build_class_weights(y_train, common_classes, args.common_weight_multiplier, args.max_class_weight)

#     save_checkpoint(args.checkpoint_dir, "checkpoint_1_split_and_weights", {
#         "rows_total": int(len(df)),
#         "rows_eval": int(len(eval_df)),
#         "classes_total": int(df[target_col].nunique()),
#         "classes_eval": int(y.nunique()),
#         "common_classes_count": int(len(common_classes)),
#         "train_rows": int(len(x_train)),
#         "test_rows": int(len(x_test)),
#     })

#     log_step("Applying dropout augmentation", f"copies={args.dropout_copies}, rate={args.dropout_rate}")
#     x_train_aug, y_train_aug = augment_with_symptom_dropout(x_train, y_train, seed=args.seed, dropout_rate=args.dropout_rate, copies=args.dropout_copies)

#     log_step("Training model", "SGDClassifier(log_loss)")
#     model = SGDClassifier(loss="log_loss", random_state=args.seed, max_iter=2, tol=1e-3, class_weight=class_weights)
#     model.fit(x_train_aug, y_train_aug)
#     log_step("Training complete")

#     os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
#     joblib.dump({"model": model, "feature_cols": feature_cols, "target_col": target_col, "common_classes": sorted(list(common_classes)), "class_weights": class_weights, "config": vars(args)}, args.model_path)
#     log_step("Model saved", args.model_path)

#     save_checkpoint(args.checkpoint_dir, "checkpoint_2_training_complete", {
#         "training_rows_after_augmentation": int(len(x_train_aug)),
#         "model_path": args.model_path,
#     })

#     log_step("Building class symptom profiles")
#     class_profiles = build_class_symptom_profiles(x_train, y_train, feature_cols)

#     log_step("Running evaluation predictions")
#     prob_raw = model.predict_proba(x_test)
#     prob_adjusted = apply_common_bias(prob_raw, model.classes_, common_classes, args.inference_common_boost)

#     print("Dataset:", args.input)
#     print("Rows used for evaluation:", len(eval_df), "/", len(df))
#     print("Classes used for evaluation:", y.nunique())
#     print("Common classes (by threshold):", len(common_classes))
#     print("Training rows after dropout augmentation:", len(x_train_aug))

#     print_metrics(y_test, prob_raw, prob_adjusted, model.classes_)

#     save_checkpoint(args.checkpoint_dir, "checkpoint_3_evaluation_complete", {
#         "raw_top1_accuracy": float(accuracy_score(y_test, model.classes_[np.argmax(prob_raw, axis=1)])),
#         "adjusted_top1_accuracy": float(accuracy_score(y_test, model.classes_[np.argmax(prob_adjusted, axis=1)])),
#     })

#     if args.row_index >= 0:
#         if args.row_index >= len(df):
#             raise IndexError(f"row-index {args.row_index} is out of bounds for dataset size {len(df)}")
#         row_x = df.iloc[[args.row_index]][feature_cols]
#         prob_row = apply_common_bias(model.predict_proba(row_x), model.classes_, common_classes, args.inference_common_boost)[0]
#         preds = top_k_predictions(prob_row, model.classes_, args.top_k)
#         print(f"\nPrediction for row-index={args.row_index}:")
#         show_prediction_output(preds, args.near_margin)

#     if args.symptoms.strip():
#         log_step("Running custom symptom inference")
#         input_symptoms = parse_symptom_input(args.symptoms)
#         row_x = build_feature_vector(feature_cols, input_symptoms)
#         prob_row = apply_common_bias(model.predict_proba(row_x), model.classes_, common_classes, args.inference_common_boost)[0]
#         reported_idx = symptom_indices(feature_cols, input_symptoms)
#         if args.positive_only_blend > 0:
#             pos_only = positive_only_match_distribution(reported_idx, model.classes_, class_profiles)
#             blend = np.clip(args.positive_only_blend, 0.0, 1.0)
#             prob_row = (1.0 - blend) * prob_row + blend * pos_only
#             prob_row = prob_row / np.sum(prob_row)
#         preds = top_k_predictions(prob_row, model.classes_, args.top_k)
#         print("\nPrediction for custom symptom input:")
#         print("Symptoms:", ", ".join(input_symptoms))
#         show_prediction_output(preds, args.near_margin)

#     log_step("Run complete")


# if __name__ == "__main__":
#     main()


import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PRIORITY_DISEASES = {
    "common cold",
    "flu",
    "pneumonia",
    "acute bronchitis",
    "asthma",
    "chronic obstructive pulmonary disease (copd)",
    "seasonal allergies (hay fever)",
    "anxiety",
    "depression",
    "diabetes",
    "infectious gastroenteritis",
    "chronic constipation",
    "gastritis",
    "gastroesophageal reflux disease (gerd)",
    "urinary tract infection",
    "eczema",
    "psoriasis",
    "acne",
    "fungal infection of the skin",
    "conjunctivitis",
    "dental caries",
    "tooth abscess",
    "gum disease",
    "ear infection (otitis media)",
    "tonsillitis",
    "strep throat",
    "migraine",
    "tension headache",
    "chronic back pain",
    "osteoarthritis",
    "kidney stone",
    "hemorrhoids",
}

PRIORITY_ALIASES = {
    "ear infection (otitis media)": "otitis media",
}


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class DiseaseClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def build_class_weights(
    y_train: pd.Series,
    common_classes: set,
    common_weight_multiplier: float,
    max_class_weight: float,
) -> Dict[str, float]:
    counts = y_train.value_counts()
    base = np.sqrt(len(y_train) / counts)
    base = base / base.mean()
    weights = base.to_dict()
    for cls in common_classes:
        if cls in weights:
            weights[cls] *= common_weight_multiplier
    for cls in list(weights.keys()):
        weights[cls] = float(min(weights[cls], max_class_weight))
    return weights


def class_weights_to_tensor(
    class_weights: Dict[str, float],
    label_encoder: LabelEncoder,
    device: torch.device,
) -> torch.Tensor:
    weight_array = np.ones(len(label_encoder.classes_), dtype=np.float32)
    for cls, w in class_weights.items():
        if cls in label_encoder.classes_:
            idx = label_encoder.transform([cls])[0]
            weight_array[idx] = w
    return torch.tensor(weight_array, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_with_symptom_dropout(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    dropout_rate: float,
    copies: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    if copies <= 0 or dropout_rate <= 0:
        return x_train, y_train
    rng = np.random.default_rng(seed)
    augmented_x_parts = [x_train]
    augmented_y_parts = [y_train]
    values = x_train.to_numpy(copy=True)
    for _ in range(copies):
        dropped = values.copy()
        ones_mask = dropped == 1
        random_mask = rng.random(dropped.shape) < dropout_rate
        dropped[np.logical_and(ones_mask, random_mask)] = 0
        augmented_x_parts.append(pd.DataFrame(dropped, columns=x_train.columns))
        augmented_y_parts.append(y_train.reset_index(drop=True))
    x_aug = pd.concat(augmented_x_parts, axis=0, ignore_index=True)
    y_aug = pd.concat(augmented_y_parts, axis=0, ignore_index=True)
    return x_aug, y_aug


# ---------------------------------------------------------------------------
# Common / priority class resolution
# ---------------------------------------------------------------------------

def find_common_classes(y: pd.Series, min_samples: int) -> set:
    counts = y.value_counts()
    return set(counts[counts >= min_samples].index)


def load_priority_classes(priority_file: str) -> set:
    if not priority_file:
        return set(PRIORITY_DISEASES)
    classes = set()
    with open(priority_file, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                classes.add(name)
    return classes


def resolve_priority_classes_in_dataset(
    all_labels: pd.Series, requested: set
) -> Tuple[set, List[str]]:
    label_set = set(all_labels.unique())
    found = set()
    missing = []
    for d in requested:
        if d in label_set:
            found.add(d)
            continue
        alias = PRIORITY_ALIASES.get(d)
        if alias and alias in label_set:
            found.add(alias)
            continue
        missing.append(d)
    return found, sorted(missing)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def apply_common_bias(
    probabilities: np.ndarray,
    classes: np.ndarray,
    common_classes: set,
    inference_common_boost: float,
) -> np.ndarray:
    class_multiplier = np.ones(len(classes), dtype=float)
    for idx, cls in enumerate(classes):
        if cls in common_classes:
            class_multiplier[idx] = inference_common_boost
    adjusted = probabilities * class_multiplier
    row_sum = adjusted.sum(axis=1, keepdims=True)
    adjusted = adjusted / np.where(row_sum == 0, 1.0, row_sum)
    return adjusted


def build_class_symptom_profiles(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
) -> Dict[str, np.ndarray]:
    train_with_target = x_train.copy()
    train_with_target["__target__"] = y_train.values
    grouped = train_with_target.groupby("__target__")[feature_cols].mean()
    return {cls: grouped.loc[cls].to_numpy(dtype=float) for cls in grouped.index}


def positive_only_match_distribution(
    reported_indices: np.ndarray,
    classes: np.ndarray,
    class_profiles: Dict[str, np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    if reported_indices.size == 0:
        return np.ones(len(classes), dtype=float) / len(classes)
    scores = np.zeros(len(classes), dtype=float)
    for i, cls in enumerate(classes):
        profile = class_profiles.get(str(cls))
        if profile is None:
            scores[i] = eps
            continue
        probs = np.clip(profile[reported_indices], eps, 1.0)
        scores[i] = float(np.exp(np.mean(np.log(probs))))
    total = scores.sum()
    if total <= 0:
        return np.ones(len(classes), dtype=float) / len(classes)
    return scores / total


def predict_proba_torch(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    all_probs = []
    tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start: start + batch_size].to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_metrics(
    y_true: pd.Series,
    prob_raw: np.ndarray,
    prob_adjusted: np.ndarray,
    classes: np.ndarray,
) -> None:
    pred_raw = classes[np.argmax(prob_raw, axis=1)]
    pred_adjusted = classes[np.argmax(prob_adjusted, axis=1)]
    top3_raw = top_k_accuracy_score(y_true, prob_raw, k=3, labels=classes)
    top3_adjusted = top_k_accuracy_score(y_true, prob_adjusted, k=3, labels=classes)
    print("\nEvaluation Metrics")
    print("-" * 80)
    print(f"Raw top-1 accuracy:          {accuracy_score(y_true, pred_raw):.4f}")
    print(f"Raw top-3 accuracy:          {top3_raw:.4f}")
    print(f"Raw macro F1:                {f1_score(y_true, pred_raw, average='macro', zero_division=0):.4f}")
    print(f"\nCommon-aware top-1 accuracy: {accuracy_score(y_true, pred_adjusted):.4f}")
    print(f"Common-aware top-3 accuracy: {top3_adjusted:.4f}")
    print(f"Common-aware macro F1:       {f1_score(y_true, pred_adjusted, average='macro', zero_division=0):.4f}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def top_k_predictions(
    prob_row: np.ndarray,
    classes: np.ndarray,
    k: int,
) -> List[Tuple[str, float]]:
    top_idx = np.argsort(prob_row)[::-1][:k]
    return [(str(classes[i]), float(prob_row[i])) for i in top_idx]


def parse_symptom_input(symptom_text: str) -> List[str]:
    return [s.strip().lower() for s in symptom_text.split(",") if s.strip()]


def build_feature_vector(feature_cols: List[str], symptoms: List[str]) -> pd.DataFrame:
    feature_set = set(symptoms)
    row = {col: 1 if col.lower() in feature_set else 0 for col in feature_cols}
    return pd.DataFrame([row])


def symptom_indices(feature_cols: List[str], symptoms: List[str]) -> np.ndarray:
    feature_map = {c.lower(): i for i, c in enumerate(feature_cols)}
    idx = [feature_map[s] for s in symptoms if s in feature_map]
    return np.array(idx, dtype=int)


def show_prediction_output(preds: List[Tuple[str, float]], near_margin: float) -> None:
    print("\nTop predictions (common-aware)")
    print("-" * 80)
    for rank, (label, conf) in enumerate(preds, start=1):
        print(f"{rank}. {label} | confidence={conf:.4f}")
    if len(preds) >= 2 and (preds[0][1] - preds[1][1]) <= near_margin:
        print(
            f"\nNear spot-on case: top-2 are close (margin <= {near_margin:.3f}). "
            "Use both as high-priority candidates."
        )


# ---------------------------------------------------------------------------
# Logging / checkpointing
# ---------------------------------------------------------------------------

def log_step(step: str, detail: str = "") -> None:
    now = datetime.now().strftime("%H:%M:%S")
    if detail:
        print(f"[{now}] {step} | {detail}", flush=True)
    else:
        print(f"[{now}] {step}", flush=True)


def save_checkpoint(checkpoint_dir: str, name: str, payload: Dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log_step("Checkpoint saved", path)


def gpu_runtime_note(device: torch.device) -> None:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        log_step("GPU detected", f"Training on: {name}")
    else:
        has_nvidia = shutil.which("nvidia-smi") is not None
        if has_nvidia:
            log_step("GPU warning", "NVIDIA found but CUDA unavailable — check PyTorch CUDA install.")
        else:
            log_step("GPU note", "No NVIDIA runtime detected — running CPU path.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Common-first disease predictor (GPU, PyTorch).")
    parser.add_argument("--input", default="sympdis_clean_basic.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--common-min-samples", type=int, default=500)
    parser.add_argument("--common-source", choices=["priority", "frequency"], default="priority")
    parser.add_argument("--priority-file", default="")
    parser.add_argument("--common-weight-multiplier", type=float, default=2.6,
                        help="Priority-class multiplier (applied only when --common-source=priority)")
    parser.add_argument("--inference-common-boost", type=float, default=1.35,
                        help="Priority-class inference boost (applied only when --common-source=priority)")
    parser.add_argument("--max-class-weight", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--near-margin", type=float, default=0.05)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--dropout-copies", type=int, default=1)
    parser.add_argument("--positive-only-blend", type=float, default=0.35)
    parser.add_argument("--symptoms", default="")
    parser.add_argument("--row-index", type=int, default=-1)
    parser.add_argument("--checkpoint-dir", default="model_checkpoints")
    parser.add_argument("--model-path", default="model_checkpoints/common_first_model.pt")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size for GPU training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_step("Run started")
    gpu_runtime_note(device)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log_step("Loading dataset", args.input)
    df = pd.read_csv(args.input)
    if df.shape[1] < 2:
        raise ValueError("Dataset must have target + features.")

    target_col = df.columns[0]
    feature_cols = list(df.columns[1:])

    log_step("Preparing evaluation dataset", "Removing singleton classes")
    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= 2].index
    eval_df = df[df[target_col].isin(valid_classes)].copy()

    x = eval_df[feature_cols]
    y = eval_df[target_col]

    log_step("Splitting train/test", f"test_size={args.test_size}")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # ------------------------------------------------------------------
    # Common / priority classes
    # ------------------------------------------------------------------
    if args.common_source == "priority":
        requested_priority = load_priority_classes(args.priority_file)
        common_classes, missing_priority = resolve_priority_classes_in_dataset(y_train, requested_priority)
        log_step("Common source", "priority list")
        log_step("Priority diseases found in training labels", str(len(common_classes)))
        if missing_priority:
            log_step("Priority diseases not found", str(len(missing_priority)))
            print("Missing priority disease labels:")
            for name in missing_priority:
                print(f"- {name}")
    else:
        common_classes = find_common_classes(y_train, args.common_min_samples)
        log_step("Common source", f"frequency threshold >= {args.common_min_samples}")

    if not common_classes:
        raise ValueError("No common/priority diseases resolved. Check labels or priority file.")

    effective_weight_multiplier = args.common_weight_multiplier if args.common_source == "priority" else 1.0
    effective_inference_boost = args.inference_common_boost if args.common_source == "priority" else 1.0

    class_weights = build_class_weights(
        y_train, common_classes, effective_weight_multiplier, args.max_class_weight
    )

    save_checkpoint(args.checkpoint_dir, "checkpoint_1_split_and_weights", {
        "rows_total": int(len(df)),
        "rows_eval": int(len(eval_df)),
        "classes_total": int(df[target_col].nunique()),
        "classes_eval": int(y.nunique()),
        "common_classes_count": int(len(common_classes)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    })

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------
    log_step("Applying dropout augmentation", f"copies={args.dropout_copies}, rate={args.dropout_rate}")
    x_train_aug, y_train_aug = augment_with_symptom_dropout(
        x_train, y_train, seed=args.seed, dropout_rate=args.dropout_rate, copies=args.dropout_copies
    )
    log_step("Augmentation complete", f"training rows = {len(x_train_aug)}")

    # ------------------------------------------------------------------
    # Label encode
    # ------------------------------------------------------------------
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train_aug)
    y_test_enc = label_encoder.transform(y_test)
    classes = label_encoder.classes_
    num_classes = len(classes)

    # ------------------------------------------------------------------
    # Build tensors and DataLoader
    # ------------------------------------------------------------------
    x_train_np = x_train_aug.to_numpy(dtype=np.float32)
    x_test_np = x_test.to_numpy(dtype=np.float32)

    train_dataset = TensorDataset(
        torch.tensor(x_train_np, dtype=torch.float32),
        torch.tensor(y_train_enc, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    model = DiseaseClassifier(input_dim=len(feature_cols), num_classes=num_classes).to(device)
    weight_tensor = class_weights_to_tensor(class_weights, label_encoder, device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_step("Training model", f"epochs={args.epochs}, batch_size={args.batch_size}, device={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)
        scheduler.step()
        avg_loss = total_loss / total
        train_acc = correct / total
        log_step(f"Epoch {epoch:>3}/{args.epochs}", f"loss={avg_loss:.4f}  train_acc={train_acc:.4f}")

    log_step("Training complete")

    # ------------------------------------------------------------------
    # Symptom profiles (for positive-only blend)
    # ------------------------------------------------------------------
    log_step("Building class symptom profiles")
    class_profiles = build_class_symptom_profiles(x_train, y_train, feature_cols)

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    bundle = {
        "model_state": model.state_dict(),
        "label_encoder": label_encoder,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "common_classes": sorted(list(common_classes)),
        "class_weights": class_weights,
        "class_profiles": class_profiles,
        "input_dim": len(feature_cols),
        "num_classes": num_classes,
        "config": vars(args),
    }
    joblib.dump(bundle, args.model_path)
    log_step("Model saved", args.model_path)

    save_checkpoint(args.checkpoint_dir, "checkpoint_2_training_complete", {
        "training_rows_after_augmentation": int(len(x_train_aug)),
        "model_path": args.model_path,
        "epochs": args.epochs,
    })

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    log_step("Running evaluation predictions")
    prob_raw = predict_proba_torch(model, x_test_np, device, batch_size=args.batch_size)
    prob_adjusted = apply_common_bias(prob_raw, classes, common_classes, effective_inference_boost)

    print("\nDataset:", args.input)
    print("Rows used for evaluation:", len(eval_df), "/", len(df))
    print("Classes used for evaluation:", y.nunique())
    print("Common classes:", len(common_classes))
    print("Training rows after dropout augmentation:", len(x_train_aug))

    print_metrics(y_test, prob_raw, prob_adjusted, classes)

    save_checkpoint(args.checkpoint_dir, "checkpoint_3_evaluation_complete", {
        "raw_top1_accuracy": float(accuracy_score(y_test, classes[np.argmax(prob_raw, axis=1)])),
        "adjusted_top1_accuracy": float(accuracy_score(y_test, classes[np.argmax(prob_adjusted, axis=1)])),
    })

    # ------------------------------------------------------------------
    # Row-index prediction
    # ------------------------------------------------------------------
    if args.row_index >= 0:
        if args.row_index >= len(df):
            raise IndexError(f"row-index {args.row_index} out of bounds for dataset size {len(df)}")
        row_x = df.iloc[[args.row_index]][feature_cols].to_numpy(dtype=np.float32)
        prob_row = predict_proba_torch(model, row_x, device)[0]
        prob_row = apply_common_bias(prob_row[np.newaxis], classes, common_classes, effective_inference_boost)[0]
        preds = top_k_predictions(prob_row, classes, args.top_k)
        print(f"\nPrediction for row-index={args.row_index}:")
        show_prediction_output(preds, args.near_margin)

    # ------------------------------------------------------------------
    # Custom symptom inference
    # ------------------------------------------------------------------
    if args.symptoms.strip():
        log_step("Running custom symptom inference")
        input_symptoms = parse_symptom_input(args.symptoms)
        row_x = build_feature_vector(feature_cols, input_symptoms).to_numpy(dtype=np.float32)
        prob_row = predict_proba_torch(model, row_x, device)[0]
        prob_row = apply_common_bias(prob_row[np.newaxis], classes, common_classes, effective_inference_boost)[0]

        reported_idx = symptom_indices(feature_cols, input_symptoms)
        if args.positive_only_blend > 0:
            pos_only = positive_only_match_distribution(reported_idx, classes, class_profiles)
            blend = np.clip(args.positive_only_blend, 0.0, 1.0)
            prob_row = (1.0 - blend) * prob_row + blend * pos_only
            prob_row = prob_row / np.sum(prob_row)

        preds = top_k_predictions(prob_row, classes, args.top_k)
        print("\nPrediction for custom symptom input:")
        print("Symptoms:", ", ".join(input_symptoms))
        show_prediction_output(preds, args.near_margin)

    log_step("Run complete")


if __name__ == "__main__":
    main()