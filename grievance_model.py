
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("🚀 Grievance Model Loader")

def train_and_save_model(csv_file):
    df = pd.read_csv(csv_file)
    X = df["Grievance"]
    y = df["Urgency"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    joblib.dump((model, vectorizer), "grievance_model.pkl")
    print("✅ Model trained and saved as grievance_model.pkl")

if __name__ == "__main__":
    file = input("Enter your CSV file name: ")
    train_and_save_model(file)
