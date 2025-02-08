from app.models import load_model

# Load the trained model
model, vectorizer = load_model()

# Loads existing model
def get_model():
    return model, vectorizer
