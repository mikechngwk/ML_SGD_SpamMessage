from fastapi import FastAPI, Depends

from app.dependencies import get_model
from app.routers.predict import router as predict_router

app = FastAPI()

# Include the router from predict.py and inject dependencies (loaded model)
app.include_router(predict_router, dependencies=[Depends(get_model)])


@app.get("/")
def root():
    return {"message": "Spam Message FastAPI is running~"}
