import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


def predict_proba_torch(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def parse_symptoms(text: str) -> list:
    return [s.strip().lower() for s in text.split(",") if s.strip()]


def build_feature_vector(feature_cols: list, symptoms: list) -> tuple:
    feature_map = {c.lower(): i for i, c in enumerate(feature_cols)}
    x = np.zeros((1, len(feature_cols)), dtype=np.float32)

    recognized = []
    unknown = []
    for s in symptoms:
        idx = feature_map.get(s)
        if idx is None:
            unknown.append(s)
        else:
            x[0, idx] = 1.0
            recognized.append(s)

    return x, recognized, unknown


def symptom_indices(feature_cols: list, symptoms: list) -> np.ndarray:
    feature_map = {c.lower(): i for i, c in enumerate(feature_cols)}
    idx = [feature_map[s] for s in symptoms if s in feature_map]
    return np.array(idx, dtype=int)


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


def build_class_symptom_profiles_from_csv(data_path: str, feature_cols: list) -> dict:
    if not data_path or not os.path.exists(data_path):
        return {}

    df = pd.read_csv(data_path)
    if df.shape[1] < 2:
        return {}

    target_col = df.columns[0]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        return {}

    grouped = df.groupby(target_col)[feature_cols].mean()
    return {cls: grouped.loc[cls].to_numpy(dtype=float) for cls in grouped.index}


def positive_only_match_distribution(
    reported_indices: np.ndarray,
    classes: np.ndarray,
    class_profiles: dict,
    eps: float = 1e-6,
) -> np.ndarray:
    if reported_indices.size == 0 or not class_profiles:
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


def top_k_predictions(prob_row: np.ndarray, classes: np.ndarray, k: int) -> list:
    top_idx = np.argsort(prob_row)[::-1][:k]
    return [(str(classes[i]), float(prob_row[i])) for i in top_idx]


def print_predictions(preds: list, near_margin: float) -> None:
    print("\nTop predictions")
    print("-" * 60)
    for i, (label, conf) in enumerate(preds, start=1):
        print(f"{i}. {label} | confidence={conf:.4f}")

    if len(preds) >= 2 and (preds[0][1] - preds[1][1]) <= near_margin:
        print(
            f"\nNear case: top-2 are close (margin <= {near_margin:.3f}). "
            "Treat both as high-priority candidates."
        )


def run_one_query(
    symptom_text: str,
    model: nn.Module,
    feature_cols: list,
    classes: np.ndarray,
    common_classes: set,
    boost: float,
    blend: float,
    class_profiles: dict,
    top_k: int,
    near_margin: float,
    device: torch.device,
) -> None:
    symptoms = parse_symptoms(symptom_text)
    x, recognized, unknown = build_feature_vector(feature_cols, symptoms)

    if len(recognized) == 0:
        print("No known symptoms recognized from input. Check spelling against dataset feature names.")
        if unknown:
            print("Unknown symptoms:", ", ".join(unknown))
        return

    prob_raw = predict_proba_torch(model, x, device)
    prob_adj = apply_common_bias(prob_raw, classes, common_classes, boost)[0]

    if blend > 0 and recognized and class_profiles:
        reported_idx = symptom_indices(feature_cols, recognized)
        if reported_idx.size > 0:
            pos_only = positive_only_match_distribution(reported_idx, classes, class_profiles)
            prob_adj = (1.0 - blend) * prob_adj + blend * pos_only
            prob_adj = prob_adj / prob_adj.sum()

    preds = top_k_predictions(prob_adj, classes, top_k)

    print("\nRecognized symptoms:", ", ".join(recognized))
    if unknown:
        print("Unknown symptoms:", ", ".join(unknown))

    print_predictions(preds, near_margin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test trained disease model v2 with custom symptom inputs.")
    parser.add_argument("--model-path", default="model_checkpoints/common_first_model_v2.pt", help="Path to saved model bundle")
    parser.add_argument("--symptoms", default="", help="Comma-separated symptom names")
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to show")
    parser.add_argument("--near-margin", type=float, default=0.05, help="Top-1/top-2 margin for near-case note")
    parser.add_argument("--inference-common-boost", type=float, default=None, help="Override common-disease boost")
    parser.add_argument("--positive-only-blend", type=float, default=0.30, help="Blend with symptom-profile score to keep rankings stable")
    parser.add_argument("--data-path", default="sympdis_clean_basic.csv", help="Cleaned dataset used to rebuild symptom profiles")
    parser.add_argument("--interactive", action="store_true", help="Run interactive multi-query mode")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    bundle = joblib.load(args.model_path)
    input_dim = int(bundle["input_dim"])
    num_classes = int(bundle["num_classes"])

    model = DiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()

    feature_cols = bundle["feature_cols"]
    label_encoder = bundle["label_encoder"]
    classes = label_encoder.classes_
    common_classes = set(bundle.get("common_classes", []))

    default_boost = bundle.get("config", {}).get("inference_common_boost", 1.0)
    boost = float(default_boost if args.inference_common_boost is None else args.inference_common_boost)
    blend = float(np.clip(args.positive_only_blend, 0.0, 1.0))
    class_profiles = bundle.get("class_profiles") or build_class_symptom_profiles_from_csv(args.data_path, feature_cols)

    print("Model loaded from:", args.model_path)
    print("Device:", device)
    print("Classes:", len(classes))
    print("Common-priority classes:", len(common_classes))
    print("Common boost:", boost)
    print("Positive-only blend:", blend)
    print("Symptom profiles available:", bool(class_profiles))

    if args.symptoms.strip():
        run_one_query(
            symptom_text=args.symptoms,
            model=model,
            feature_cols=feature_cols,
            classes=classes,
            common_classes=common_classes,
            boost=boost,
            blend=blend,
            class_profiles=class_profiles,
            top_k=args.top_k,
            near_margin=args.near_margin,
            device=device,
        )

    if args.interactive:
        print("\nInteractive mode: type symptoms separated by commas. Type exit to quit.")
        while True:
            text = input("\nSymptoms> ").strip()
            if text.lower() in {"exit", "quit", "q"}:
                break
            if not text:
                continue
            run_one_query(
                symptom_text=text,
                model=model,
                feature_cols=feature_cols,
                classes=classes,
                common_classes=common_classes,
                boost=boost,
                blend=blend,
                class_profiles=class_profiles,
                top_k=args.top_k,
                near_margin=args.near_margin,
                device=device,
            )

    if not args.interactive and not args.symptoms.strip():
        print("No query provided. Use --symptoms or --interactive.")


if __name__ == "__main__":
    main()
