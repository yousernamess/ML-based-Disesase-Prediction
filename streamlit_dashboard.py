import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
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


def symptom_indices(feature_cols: list, symptoms: list) -> np.ndarray:
    feature_map = {c.lower(): i for i, c in enumerate(feature_cols)}
    idx = [feature_map[s] for s in symptoms if s in feature_map]
    return np.array(idx, dtype=int)


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


def build_feature_vector(feature_cols: list, symptoms: list) -> tuple:
    feature_map = {c.lower(): i for i, c in enumerate(feature_cols)}
    x = np.zeros((1, len(feature_cols)), dtype=np.float32)

    recognized = []
    unknown = []
    for s in symptoms:
        key = s.strip().lower()
        if not key:
            continue
        idx = feature_map.get(key)
        if idx is None:
            unknown.append(s)
        else:
            x[0, idx] = 1.0
            recognized.append(key)

    return x, recognized, unknown


def predict_proba_torch(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


@st.cache_resource(show_spinner=False)
def load_bundle(model_path: str, device_name: str):
    bundle = joblib.load(model_path)

    input_dim = int(bundle["input_dim"])
    num_classes = int(bundle["num_classes"])

    device = torch.device(device_name)
    model = DiseaseClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()

    feature_cols = bundle["feature_cols"]
    label_encoder = bundle["label_encoder"]
    classes = label_encoder.classes_
    common_classes = set(bundle.get("common_classes", []))
    default_boost = float(bundle.get("config", {}).get("inference_common_boost", 1.0))
    class_profiles = bundle.get("class_profiles", {})

    return model, feature_cols, classes, common_classes, default_boost, class_profiles, device


def top_k_predictions(prob_row: np.ndarray, classes: np.ndarray, k: int):
    top_idx = np.argsort(prob_row)[::-1][:k]
    return [(str(classes[i]), float(prob_row[i])) for i in top_idx]


def main() -> None:
    st.set_page_config(page_title="Symptom to Disease", layout="wide")
    st.title("Symptom to Disease Predictor")
    st.caption("Top-k disease predictions with adjustable common-priority bias")

    if "last_prediction_vector" not in st.session_state:
        st.session_state.last_prediction_vector = None
        st.session_state.last_model_path = None

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model Path", value="model_checkpoints/common_first_model_v2.pt")
        data_path = st.text_input("Cleaned data path", value="sympdis_clean_basic.csv")

        if st.checkbox("Force CPU", value=False):
            device_name = "cpu"
        else:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        top_k = st.slider("Top-k diseases", min_value=1, max_value=10, value=3, step=1)
        near_margin = st.slider("Near-case margin", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
        positive_only_blend = st.slider("Symptom-profile blend", min_value=0.0, max_value=0.80, value=0.30, step=0.05)
        stability_carryover = st.slider("Prediction stability", min_value=0.0, max_value=0.75, value=0.35, step=0.05)
        show_raw_probs = st.checkbox("Show raw probabilities", value=False)
        clear_memory = st.button("Clear prediction memory")

    try:
        model, feature_cols, classes, common_classes, default_boost, class_profiles, device = load_bundle(model_path, device_name)
    except Exception as exc:
        st.error(f"Failed to load model bundle: {exc}")
        st.stop()

    if clear_memory:
        st.session_state.last_prediction_vector = None
        st.session_state.last_model_path = None
        st.rerun()

    if st.session_state.last_model_path != model_path:
        st.session_state.last_prediction_vector = None
        st.session_state.last_model_path = model_path

    if not class_profiles:
        class_profiles = build_class_symptom_profiles_from_csv(data_path, feature_cols)

    st.success(f"Model loaded on {device} | Classes: {len(classes)} | Priority classes: {len(common_classes)}")

    boost = st.slider(
        "Common-priority boost",
        min_value=1.0,
        max_value=2.0,
        value=float(default_boost),
        step=0.05,
    )

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("Input Symptoms")
        selected_symptoms = st.multiselect(
            "Pick symptoms from dataset features",
            options=feature_cols,
            default=[],
        )

        typed_symptoms = st.text_area(
            "Or type comma-separated symptoms",
            value="",
            placeholder="fever, cough, headache",
        )

        run = st.button("Predict", type="primary")

    with col_b:
        st.subheader("Input Summary")
        st.write(f"Feature count in model: {len(feature_cols)}")
        st.write(f"Priority classes in model: {len(common_classes)}")

    if not run:
        st.info("Select or type symptoms, then click Predict.")
        return

    typed_list = [s.strip() for s in typed_symptoms.split(",") if s.strip()]
    all_symptoms = list(dict.fromkeys(selected_symptoms + typed_list))

    x, recognized, unknown = build_feature_vector(feature_cols, all_symptoms)

    if len(recognized) == 0:
        st.warning("No recognized symptoms found. Check symptom spelling.")
        if unknown:
            st.write("Unknown symptoms:", unknown)
        return

    prob_raw = predict_proba_torch(model, x, device)
    prob_adj = apply_common_bias(prob_raw, classes, common_classes, boost)

    if positive_only_blend > 0 and recognized and class_profiles:
        reported_idx = symptom_indices(feature_cols, recognized)
        if reported_idx.size > 0:
            pos_only = positive_only_match_distribution(reported_idx, classes, class_profiles)
            prob_adj = (1.0 - positive_only_blend) * prob_adj + positive_only_blend * pos_only
            prob_adj = prob_adj / prob_adj.sum()

    previous_vector = st.session_state.last_prediction_vector
    if previous_vector is not None and len(previous_vector) == len(prob_adj[0]) and stability_carryover > 0:
        prob_adj[0] = (1.0 - stability_carryover) * prob_adj[0] + stability_carryover * previous_vector
        prob_adj[0] = prob_adj[0] / prob_adj[0].sum()

    preds = top_k_predictions(prob_adj[0], classes, top_k)

    st.session_state.last_prediction_vector = prob_adj[0].copy()

    st.subheader("Prediction Results")
    result_rows = []
    for rank, (disease, conf) in enumerate(preds, start=1):
        result_rows.append({
            "rank": rank,
            "disease": disease,
            "confidence": conf,
            "confidence_%": conf * 100,
        })

    st.dataframe(result_rows, use_container_width=True)

    if len(preds) >= 2 and (preds[0][1] - preds[1][1]) <= near_margin:
        st.warning(
            f"Near case: top-2 are close (margin <= {near_margin:.2f}). "
            "Treat both as high-priority candidates."
        )

    st.write("Recognized symptoms:", recognized)
    if unknown:
        st.write("Unknown symptoms:", unknown)

    if show_raw_probs:
        raw_preds = top_k_predictions(prob_raw[0], classes, top_k)
        st.subheader("Raw (Unbiased) Top-k")
        raw_rows = []
        for rank, (disease, conf) in enumerate(raw_preds, start=1):
            raw_rows.append({
                "rank": rank,
                "disease": disease,
                "confidence": conf,
                "confidence_%": conf * 100,
            })
        st.dataframe(raw_rows, use_container_width=True)


if __name__ == "__main__":
    main()
