from fastapi import APIRouter, Depends
from app.main import get_model

router = APIRouter()


@router.get("/predict")
def predict_spam(text: str, model_data=Depends(get_model)):
    """
    Predict if a given text is spam or not.
    1. Vectorize the input text
    2. Get the prediction based on the vectorized_input (Step1)
    3. Print the probabilities (NO-SPAM/SPAM)
    4. Print the probabilities (NO-SPAM/SPAM) in JSON
    """

    model, vectorizer = model_data  # Get model and vectorizer via dependency injection from main.py

    vectorized_input = vectorizer.transform([text])
    print(f"Vectorized input for '{text}':\n{vectorized_input.toarray()}")  # Show the vectorized input

    prediction = model.predict(vectorized_input)
    print(f"Prediction: {prediction}")

    probabilities = model.predict_proba(vectorized_input)
    print(f"Prediction probabilities: {probabilities}")

    return {
        "prediction": int(prediction[0]),
        "probabilities": probabilities.tolist()  # Return the probabilities for further analysis
    }
