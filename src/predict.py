import joblib
import pandas as pd

model = joblib.load("models/severity_model.pkl")

def predict_accident(vehiculos, mes, fin_semana, tipo_accidente):
    
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

    return prediction, probability


if __name__ == "__main__":
    
    pred, prob = predict_accident(
        vehiculos=3,
        mes=12,
        fin_semana=1,
        tipo_accidente="CHOQUE"
    )

    print("Predicción:", pred)
    print("Probabilidad de fatalidad:", prob)