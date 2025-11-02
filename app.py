# pyright: reportMissingImports=false
import os
import sys
import numpy as np  # type: ignore[reportMissingImports]
import joblib  # type: ignore[reportMissingImports]
import gradio as gr  # type: ignore[reportMissingImports]
from sklearn.pipeline import make_pipeline  # type: ignore[reportMissingImports]


# Compatibility shim for models pickled with a custom TextPreprocessor defined in __main__
class TextPreprocessor:  # noqa: N801 - keep original class name for pickle compatibility
    def __init__(self, **kwargs):
        self.params = dict(kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return [self._clean_text(t) for t in X]
        except TypeError:
            return [self._clean_text(X)]

    def _clean_text(self, text):
        if text is None:
            return ""
        # Minimal, safe normalization; original training may differ
        return str(text).strip()

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        return self

# Also register the shim on the __main__ module so unpickling works when this file is imported
try:
    setattr(sys.modules.get("__main__"), "TextPreprocessor", TextPreprocessor)
except Exception:
    pass


# Provide additional aliases and a generic fallback for other custom transformers
class FeatureExtractor(TextPreprocessor):  # common alias observed in notebooks
    pass

try:
    main_mod = sys.modules.get("__main__")
    if main_mod is not None:
        setattr(main_mod, "FeatureExtractor", FeatureExtractor)

        def __getattr__(name):  # type: ignore
            # Fallback for any other unknown custom class names in the pickle
            return TextPreprocessor

        setattr(main_mod, "__getattr__", __getattr__)
except Exception:
    pass


def _get_model_path() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "grievance_model.pkl")


LABEL_ENCODER = None  # Optional: discovered LabelEncoder to map indices -> human labels


def _find_label_encoder(obj):
    """Recursively search for a LabelEncoder-like object with inverse_transform and classes_."""
    try:
        # Direct match
        if hasattr(obj, "inverse_transform") and hasattr(obj, "classes_"):
            return obj
        # Walk dicts, lists, tuples, objects
        if isinstance(obj, dict):
            for v in obj.values():
                enc = _find_label_encoder(v)
                if enc is not None:
                    return enc
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                enc = _find_label_encoder(v)
                if enc is not None:
                    return enc
        elif hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__"), dict):
            for v in obj.__dict__.values():
                enc = _find_label_encoder(v)
                if enc is not None:
                    return enc
    except Exception:
        pass
    return None


def load_model():
    model_path = _get_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Ensure grievance_model.pkl exists."
        )
    loaded = joblib.load(model_path)
    # Try to discover a LabelEncoder to map class indices back to names
    try:
        global LABEL_ENCODER
        LABEL_ENCODER = _find_label_encoder(loaded)
    except Exception:
        pass

    def pick_components(obj):
        # Returns (model, vectorizer, preprocessor)
        model_candidate = None
        vectorizer_candidate = None
        preprocessor_candidate = None

        # Helper utilities to robustly search nested structures
        def is_model(x):
            return hasattr(x, "predict") or hasattr(x, "steps")

        def looks_like_vectorizer(x):
            if not hasattr(x, "transform") or hasattr(x, "predict"):
                return False
            type_name = type(x).__name__.lower()
            return (
                "vectorizer" in type_name
                or hasattr(x, "vocabulary_")
                or hasattr(x, "get_feature_names_out")
            )

        def looks_like_preprocessor(x):
            return hasattr(x, "transform") and not hasattr(x, "predict") and not looks_like_vectorizer(x)

        def walk_candidates(x):
            try:
                if hasattr(x, "steps"):
                    for _, step in getattr(x, "steps", []) or []:
                        yield step
                # Inspect estimator/object attributes for nested transformers
                if hasattr(x, "__dict__") and isinstance(getattr(x, "__dict__"), dict):
                    for v in x.__dict__.values():
                        yield v
                if isinstance(x, dict):
                    for v in x.values():
                        yield v
                elif isinstance(x, (list, tuple)):
                    for v in x:
                        yield v
            except Exception:
                pass

        # Case 1: explicit pair (tuple or list)
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return obj[0], obj[1], None

        # Case 2: sklearn-like object or pipeline
        if is_model(obj):
            if hasattr(obj, "steps"):
                return obj, None, None
            # Try discover associated transformers nested alongside
            for candidate in walk_candidates(obj):
                if looks_like_vectorizer(candidate):
                    return obj, candidate, None
                if preprocessor_candidate is None and looks_like_preprocessor(candidate):
                    preprocessor_candidate = candidate
            return obj, None, preprocessor_candidate

        # Case 3: mapping/dict
        if isinstance(obj, dict):
            # Try common keys first
            for key in ["pipeline", "model", "estimator", "clf", "classifier"]:
                if key in obj and (hasattr(obj[key], "predict") or hasattr(obj[key], "steps")):
                    model_candidate = obj[key]
                    break
            if model_candidate is None:
                # Fall back: search values heuristically
                for value in obj.values():
                    if hasattr(value, "predict") or hasattr(value, "steps"):
                        model_candidate = value
                        break
            # Vectorizer by common keys (prefer true Vectorizers over Transformers)
            for key in ["vectorizer", "tfidf_vectorizer", "tokenizer", "tfidf"]:
                if key in obj and hasattr(obj[key], "transform"):
                    val = obj[key]
                    type_name = type(val).__name__.lower()
                    # Prefer objects that look like text vectorizers
                    if (
                        "vectorizer" in type_name
                        or hasattr(val, "vocabulary_")
                        or hasattr(val, "get_feature_names_out")
                    ):
                        vectorizer_candidate = val
                        break
                    # If this looks like a transformer (e.g., TfidfTransformer), treat it as preprocessor
                    if preprocessor_candidate is None:
                        preprocessor_candidate = val
            # As a fallback, search recursively for candidates
            if vectorizer_candidate is None:
                for value in obj.values():
                    if looks_like_vectorizer(value):
                        vectorizer_candidate = value
                        break
                    if preprocessor_candidate is None and looks_like_preprocessor(value):
                        preprocessor_candidate = value
                    for nested in walk_candidates(value):
                        if looks_like_vectorizer(nested):
                            vectorizer_candidate = nested
                            break
                        if preprocessor_candidate is None and looks_like_preprocessor(nested):
                            preprocessor_candidate = nested
                    if vectorizer_candidate is not None:
                        break
            # As a final fallback, find any transformer-like object
            if vectorizer_candidate is None:
                for value in obj.values():
                    if hasattr(value, "transform") and not hasattr(value, "predict"):
                        # Prefer a likely TF-IDF Vectorizer (not Transformer) if detectable
                        type_name = type(value).__name__.lower()
                        if (
                            "vectorizer" in type_name
                            or hasattr(value, "vocabulary_")
                            or hasattr(value, "get_feature_names_out")
                        ):
                            vectorizer_candidate = value
                            break
                        if preprocessor_candidate is None:
                            preprocessor_candidate = value
                # If we still don't have a vectorizer but we have a preprocessor, swap if vectorizer missing
            if model_candidate is not None:
                return model_candidate, vectorizer_candidate, preprocessor_candidate

        # Default: treat as model only
        return obj, None, None

    model, vectorizer, preprocessor = pick_components(loaded)

    # If we have a vectorizer (and optionally a preprocessor), compose a Pipeline so
    # downstream code can pass raw text directly to MODEL.
    try:
        if hasattr(model, "steps"):
            # Already a Pipeline, no separate vectorizer needed downstream
            return model, None, None

        if vectorizer is not None and hasattr(vectorizer, "transform"):
            pipeline_steps = []
            if preprocessor is not None and hasattr(preprocessor, "transform"):
                pipeline_steps.append(preprocessor)
            pipeline_steps.append(vectorizer)
            pipeline_steps.append(model)
            pipeline = make_pipeline(*pipeline_steps)
            # Return the composed Pipeline as MODEL; clear vectorizer/preprocessor to avoid double-processing
            return pipeline, None, None
    except Exception:
        # Fall through to returning raw components if Pipeline composition fails
        pass

    return model, vectorizer, preprocessor


MODEL, VECTORIZER, PREPROCESSOR = load_model()


def _predict_proba_internal(text: str):
    """Return (labels, probs) if available; else (labels, probs) with a single label and prob=1.0."""
    X_raw = [text]

    # Case 1: MODEL can take raw text (Pipeline or text-accepting estimator)
    try:
        if hasattr(MODEL, "predict_proba"):
            try:
                probs = MODEL.predict_proba(X_raw)[0]
                classes = getattr(MODEL, "classes_", None)
                if classes is None and hasattr(MODEL, "steps") and len(MODEL.steps) > 0:
                    classes = getattr(MODEL.steps[-1][1], "classes_", None)
                if classes is not None:
                    decoded = _maybe_decode_labels(classes)
                    return decoded, probs
            except Exception:
                pass
    except Exception:
        pass

    # Case 2: Use vectorizer + model
    try:
        X_features = None
        if VECTORIZER is not None and hasattr(VECTORIZER, "transform"):
            X_features = VECTORIZER.transform(X_raw)
        # Probabilities
        if hasattr(MODEL, "predict_proba") and X_features is not None:
            probs = MODEL.predict_proba(X_features)[0]
            classes = getattr(MODEL, "classes_", None)
            if classes is None and hasattr(MODEL, "steps") and len(MODEL.steps) > 0:
                classes = getattr(MODEL.steps[-1][1], "classes_", None)
            if classes is not None:
                decoded = _maybe_decode_labels(classes)
                return decoded, probs
        # Fallback to label only
        predicted = MODEL.predict(X_features if X_features is not None else X_raw)[0]
        label = _maybe_decode_single(predicted)
        return [label], [1.0]
    except Exception:
        # Final fallback: try predict on raw text
        try:
            predicted = MODEL.predict(X_raw)[0]
            label = _maybe_decode_single(predicted)
            return [label], [1.0]
        except Exception:
            return [""], [1.0]


def predict_urgency(grievance_text: str):
    if not grievance_text or not grievance_text.strip():
        return {"": 1.0}

    labels, probs = _predict_proba_internal(grievance_text)
    if not labels:
        return {"": 1.0}
    return {label: float(prob) for label, prob in zip(labels, probs)}


def _maybe_decode_labels(classes):
    """Try to map numeric class indices to human-readable labels via discovered LABEL_ENCODER."""
    try:
        if LABEL_ENCODER is None:
            return [str(x) for x in classes]
        arr = np.array(classes)
        if arr.dtype.kind not in ("i", "u"):
            # Attempt to coerce to integers (e.g., strings like '0', '1')
            arr = arr.astype(int)
        decoded = LABEL_ENCODER.inverse_transform(arr)
        return [str(x) for x in decoded]
    except Exception:
        return [str(x) for x in classes]


def _maybe_decode_single(label):
    try:
        if LABEL_ENCODER is None:
            return str(label)
        val = int(label)
        decoded = LABEL_ENCODER.inverse_transform([val])[0]
        return str(decoded)
    except Exception:
        return str(label)


def predict_urgency_label(grievance_text: str) -> str:
    """Return only the best label."""
    labels, probs = _predict_proba_internal(grievance_text)
    if not labels:
        return ""
    # If we got synthetic single-label prob, return that label
    if len(labels) == 1:
        return labels[0]
    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return labels[best_idx]


def predict_urgency_topk(grievance_text: str, k: int = 3):
    """Return a sorted list of (label, prob) pairs for the top-k classes."""
    labels, probs = _predict_proba_internal(grievance_text)
    if not labels:
        return []
    pairs = list(zip(labels, map(float, probs)))
    pairs.sort(key=lambda lp: lp[1], reverse=True)
    return pairs[: max(1, k)]


def predict_advanced(grievance_text: str):
    """Return three views: dict for Label, full JSON dict, and top-1 label with confidence."""
    mapping = predict_urgency(grievance_text)
    if not mapping:
        return {"": 1.0}, {}, ""
    # full dict
    full_json = {k: float(v) for k, v in mapping.items()}
    # top-1
    top_label, top_prob = max(full_json.items(), key=lambda kv: kv[1])
    top_text = f"{top_label} ({top_prob:.3f})"
    return mapping, full_json, top_text


demo = gr.Interface(
    fn=predict_urgency,
    inputs=gr.Textbox(label="Enter grievance text", lines=6, placeholder="Describe the student's grievance here..."),
    outputs=gr.Label(label="Predicted Urgency", num_top_classes=3),
    title="Student Grievance Categorizer",
    description=(
        "Type a grievance and get the predicted urgency. "
        "This demo loads the trained RandomForest model from grievance_model.pkl."
    ),
)

# Advanced panel exposing more prediction views without altering the simple interface
advanced_demo = gr.Interface(
    fn=predict_advanced,
    inputs=gr.Textbox(label="Enter grievance text", lines=6, placeholder="Describe the student's grievance here..."),
    outputs=[
        gr.Label(label="Top classes", num_top_classes=5),
        gr.JSON(label="All class probabilities"),
        gr.Textbox(label="Best label (confidence)")
    ],
    title="Student Grievance Categorizer – Advanced",
    description=(
        "View top-5 classes, full probability distribution, and best label with confidence."
    ),
)

# Combine both into tabs while keeping the original interface available as `demo`
app = gr.TabbedInterface([demo, advanced_demo], ["Simple", "Advanced"])


if __name__ == "__main__":
    # Launch tabbed app; the original `demo` remains available programmatically for backward use
    app.launch(server_name="0.0.0.0", server_port=7860)


