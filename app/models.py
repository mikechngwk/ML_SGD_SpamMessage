import pandas as pd
import joblib
import os
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

MODEL_FILE = "spam_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATASET_FILE = "spam_data.xlsx"

def train_model():
    """
    Train the spam classifier model incrementally using an SGDClassifier.
    This function vectorized the text(messages) and trains the model,
    and saves both the trained model and vectorizer.
    """
    # Initialize the model (SGDClassifier supports incremental learning)
    model = SGDClassifier(loss='log_loss', max_iter=10000, tol=1e-3)

    # Initialize the vectorizer to convert text data into matrix of token counts
    vectorizer = CountVectorizer()

    # Check if dataset file(spam_data.xlsx) exists
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file '{DATASET_FILE}' not found!")

    df = pd.read_excel(DATASET_FILE)

    # To check if dataset has required columns for training model
    if "text" not in df.columns or "target" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'target' columns!")

    X_message = df["text"]
    y_value = df["target"]

    """
    Split dataset into:
    80% for training
    20% for testing
    This is crucial as 20% of the data needs to be "unseen" from the model and be used to test the trained model
    """
    X_train, X_val, y_train, y_val = train_test_split(X_message, y_value, test_size=0.2)

    """
    The following step fits and transforms the X_train dataset where the model learns the vocabulary and transform the training data
    """
    X_train_vectorized = vectorizer.fit_transform(X_train)
    """
    The following step used the already learned vocabulary from the step above and transforms new data
    """
    X_val_vectorized = vectorizer.transform(X_val)

    # Train model
    model.fit(X_train_vectorized, y_train)

    # Evaluate on validation data
    y_pred = model.predict(X_val_vectorized)
    print(classification_report(y_val, y_pred))

    # Save the trained model and vectorizer
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"Model and vectorizer saved: '{MODEL_FILE}', '{VECTORIZER_FILE}'")


def load_model():
    """
    Load the trained spam classifier model and vectorizer.
    If the model does not exist, it triggers training.
    """
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        print("Model and vectorizer loaded successfully!")
    else:
        print("No trained model found. Training a new model...")
        train_model()
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)

    return model, vectorizer


# If this script is run directly, it trains the model
if __name__ == "__main__":
    train_model()