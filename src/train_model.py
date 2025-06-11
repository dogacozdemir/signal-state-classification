from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print(">>> Classification results:")
    print(classification_report(y_test, preds))

def save_model(model, path="models/model.pkl"):
    # dumb model save (can overwrite old one)
    joblib.dump(model, path)
