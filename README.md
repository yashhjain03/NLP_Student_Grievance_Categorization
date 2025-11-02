# NLP_Student_Grievance_Categorization
a repository to track and share code for the nlp project Student Grievance Categorization.

## Frontend (Gradio) Quickstart

This repo now includes a simple Gradio frontend that loads `grievance_model.pkl` and predicts the urgency for a given grievance text.

### 1) Install dependencies

Recommended minimal install for the app:

```
pip install --upgrade pip
pip install gradio scikit-learn joblib numpy scipy pandas
```

Alternatively, to install everything listed in `requirements.txt` (heavier):

```
pip install -r requirements.txt
```

### 2) Run the app

```
python app.py
```

Open `http://localhost:7860` in your browser.

Notes:
- Ensure `grievance_model.pkl` exists in the project root. If not, you can create it by training via `grievance_model.py`.
- By default, the server binds to `0.0.0.0:7860` for convenience in containers/VMs.
