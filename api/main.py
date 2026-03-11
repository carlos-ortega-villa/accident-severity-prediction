from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()


modelo = joblib.load("modelo_accidentes.pkl")

class AccidenteInput(BaseModel):
    Año: int
    Mes: int
    Trimestre: int
    Dia_semana: str
    Fin_de_Semana: str
    Barrio_Agrupado: str
    Clase_de_Accidente: str

@app.get("/")
def home():
    return {"mensaje": "API de Predicción de Severidad de Accidentes"}

@app.post("/predict")
def predecir(data: AccidenteInput):
    
    df = pd.DataFrame([{
        "Año": data.Año,
        "Mes": data.Mes,
        "Trimestre": data.Trimestre,
        "Dia_semana": data.Dia_semana,
        "Fin_de_Semana": data.Fin_de_Semana,
        "Barrio_Agrupado": data.Barrio_Agrupado,
        "Clase de Accidente": data.Clase_de_Accidente
    }])
    
    prediccion = modelo.predict(df)[0]
    
    return {
        "prediccion": int(prediccion),
        "descripcion": "Fatal" if prediccion == 1 else "No Fatal"
    }