from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Accident Severity Prediction API")

model = joblib.load("models/severity_model.pkl")


@app.get("/")
def home():
    return {"message": "API de predicción de accidentes funcionando"}


@app.post("/predict")
def predict_accident(vehiculos: int, mes: int, fin_semana: int, tipo_accidente: str):

    data = pd.DataFrame({
        "Vehiculos Involucrados": [vehiculos],
        "mes": [mes],
        "fin_semana": [fin_semana],
        "Clase de Accidente": [tipo_accidente]
    })

    data = pd.get_dummies(data)

    model_columns = model.feature_names_in_

    for col in model_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[model_columns]

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "prediccion": int(prediction),
        "probabilidad_fatal": float(probability)
    }