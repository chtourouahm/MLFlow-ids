from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import sklearn
from fastapi import FastAPI, File, UploadFile
import uvicorn
import sys  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
from src.clean_data_csv import clean_data
from src.clean_data_json import clean_data_json
from models.transaction_info import TransactionModel
import os

#DagsHub_username = os.environ["DAGSHUB_USERNAME"]
#DagsHub_token= os.environ["DAGSHUB_TOKEN"]

# os.environ['MLFLOW_TRACKING_USERNAME']= "bassem.benhamed"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "bd4127ad9b5283f90b52886f5070b8d78c19bcaf"

# #setup mlflow
# mlflow.set_tracking_uri('https://dagshub.com/bassem.benhamed/idsd_sd.mlflow') #your mlfow tracking uri
# mlflow.set_experiment("idsd-sd-experiment")






app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#let's call the model from the model registry ( in production stage)

all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
df_mlflow = mlflow.search_runs(experiment_ids=all_experiments,filter_string="metrics.F1_score_test <1")
run_id = df_mlflow.loc[df_mlflow['metrics.F1_score_test'].idxmax()]['run_id']

logged_model = f'runs:/{run_id}/ML_models'

model = mlflow.pyfunc.load_model(logged_model)


@app.get("/")
def read_root():
    return {"Hello": "to fraud detector app"}

# this endpoint receives data in the form of csv file (histotical transactions data)
@app.post("/predict/csv")
def return_predictions(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    preprocessed_data = clean_data(data)
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}


# this endpoint receives data in the form of json (informations about one transaction)
@app.post("/predict")
def predict(data : TransactionModel):
    received = data.dict()
    df =  pd.DataFrame(received,index=[0])
    preprocessed_data = clean_data_json(df)
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)

