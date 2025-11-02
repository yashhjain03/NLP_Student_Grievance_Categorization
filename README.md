## Student Grievance Classification – Start to End Guide
This project classifies student grievances and estimates urgency. It includes a Gradio web UI, a minimal trainer, and a notebook for experimentation.

### What’s included
- **Web app**: `app.py` (Gradio UI with Simple and Advanced tabs)
- **Pretrained model**: `grievance_model.pkl` (required by the app)
- **Minimal trainer**: `grievance_model.py` (train on your CSV with columns `Grievance`, `Urgency`)
- **Notebook**: `grievance_model.ipynb` (for interactive training)

## Prerequisites
- Python 3.9+
- pip

## 1) Set up environment
```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
```

## 2) Install dependencies
Option A: use the provided requirements
```bash
pip install -r requirements.txt
```

Option B: minimal runtime for the app
```bash
pip install gradio scikit-learn joblib numpy scipy pandas
```

## 3) Ensure a model exists
The app expects `grievance_model.pkl` in the project root.

- If you already have it, skip to Run the app.
- Otherwise, train a quick model with the minimal trainer:

### 3.a) Prepare your CSV
Your CSV must contain at least:
- **Grievance**: text input
- **Urgency**: target label (e.g., Low/Medium/High)

Example (CSV snippet):
```csv
Grievance,Urgency
"WiFi keeps disconnecting in my dorm room",Low
"I'm unsure I can afford next semester's fees",Medium
"Emergency medical bill; cannot pay and need urgent help",High
```

### 3.b) Train and save
```bash
python grievance_model.py
# Enter your CSV file name when prompted, e.g.: dataset.csv
```
This produces `grievance_model.pkl` in the project root.

### 3.c) Optional: Train in a notebook (Colab/Jupyter)
- Open `grievance_model.ipynb`
- Run cells to train and export a `grievance_model.pkl`
- Place the generated file in the project root

## 4) Run the app
```bash
python app.py
```
Then open `http://localhost:7860`.

### Simple tab
- Type grievance text and see the predicted urgency with top probabilities.

### Advanced tab
- Shows top‑k classes and full probability distribution
- Displays best label with confidence

## 5) Programmatic usage
Import helpers directly from `app.py` after the model has loaded:
```python
from app import predict_urgency, predict_urgency_label, predict_urgency_topk

text = "WiFi keeps disconnecting in my dorm room"
probs = predict_urgency(text)                    # dict: label -> prob
label = predict_urgency_label(text)              # best label
top3  = predict_urgency_topk(text, k=3)          # list[(label, prob)]
```

## Common issues
- **Model file not found**: Ensure `grievance_model.pkl` exists in the project root.
- **Different training schema**: If your pickle includes custom preprocessors/encoders, `app.py` includes shims and attempts to decode class indices to human labels. If labels still appear as numbers, retrain with a proper `LabelEncoder` or map labels externally.
- **Port in use (7860)**: Edit the launch line in `app.py` to change `server_port`.
- **Dependency errors**: Re‑run `pip install -r requirements.txt` in a clean virtual environment.

## Notes
- The UI binds to `0.0.0.0:7860` for container/VM access. Locally, open `http://localhost:7860`.
- Place large datasets under a `data/` folder (ignored by git).
