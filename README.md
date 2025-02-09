## **Project Description**

This project demonstrates the use of machine learning to classify text messages as spam or non-spam using the SGDClassifier (Stochastic Gradient Descent Classifier) from scikit-learn. The project builds and trains a machine learning model that processes textual data, converts it into numerical features using `CountVectorizer`, and then applies a classification algorithm to predict whether a given message is spam.

## **Technologies Used**

- **Python**: The primary programming language for building the application.
- **Scikit-learn**: A machine learning library used to implement the SGDClassifier for spam message classification.
- **FastAPI**: A fast, modern web framework for building the API that serves the model predictions.
- **Uvicorn**: ASGI server for running the FastAPI application.
- **CountVectorizer**: A tool from scikit-learn used to convert text data into a numerical format for machine learning.

## **Installation Instructions**

To run the project, you’ll need to install the following dependencies. You can set up your environment and install the required packages using the `requirements.txt` file.

### 1. Set up your virtual environment (optional but recommended):

```bash
python -m venv venv
```

### 2. Activate the virtual environment:

- On **Windows**:
```bash
venv\Scripts\activate
```

- On **macOS/Linux**:
```bash
source venv/bin/activate
```
### 3. Install the dependencies:
Once the virtual environment is activated, run the following command to install all the required packages:
```bash
pip install -r requirements.txt
```

### 4. Running the Application:
After installing the dependencies, you can run the FastAPI server with the following command:
```bash
uvicorn app.main:app --reload
```
## **How to Use**
Once the server is running, you can interact with the API through the /predict endpoint to make predictions about text messages.

## **Example Usage:**
To predict whether a given text is spam, send a GET request to /predict with a query parameter text. For example:
```bash
GET http://127.0.0.1:8000/predict?text=Congratulations%20You%20won%20a%20prize!
```
## **Example Response:**
```bash
{
    "prediction": 1,
    "probabilities": [[0.2, 0.8]]
}
```

## **Example Input/Output:**
- **Input**:
  - **Text**: "Congratulations! You've won a free prize."
- **Output**:
  - **Prediction**: 1 (Spam)
  - **Probabilities**: `[[0.2, 0.8]]`


"prediction": 1, (Spam) <br>
"prediction": 0, (No-Spam) <br>
"probabilities": [[X, Y]] | X = Probability of Not Spam, Y = Probability of Spam















