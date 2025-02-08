# **ML_SGDClassifier_SpamMessages**

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
