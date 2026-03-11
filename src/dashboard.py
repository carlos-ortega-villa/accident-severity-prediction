import streamlit as st
import joblib
import pandas as pd

 
st.set_page_config(
    page_title="Predicción de Severidad de Accidentes",
    page_icon="🚗",
    layout="wide"
)

 
st.title("Sistema de Predicción de Severidad de Accidentes 🚗 ")
st.markdown(
    "Aplicación basada en **Machine Learning (Random Forest)** para estimar la probabilidad de fatalidad en accidentes de tránsito."
)

st.divider()

 
model = joblib.load("models/severity_model.pkl")



col1, col2 = st.columns(2)





with col1:

    st.subheader(" Configuración del accidente")

    vehiculos = st.slider(
        "Vehículos involucrados",
        1, 10, 2
    )

    mes = st.selectbox(
        "Mes del accidente",
        list(range(1, 13))
    )

    fin_semana = st.selectbox(
        "¿Ocurrió en fin de semana?",
        ["No", "Sí"]
    )

    tipo_accidente = st.selectbox(
        "Tipo de accidente",
        ["CHOQUE", "OTRO", "VOLCAMIENTO"]
    )



fin_semana = 1 if fin_semana == "Sí" else 0




with col2:

    st.subheader(" Resultado de la predicción")

    if st.button("Predecir severidad"):

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

        st.metric(
            label="Probabilidad de accidente fatal",
            value=f"{probability:.2%}"
        )

        if prediction == 1:
            st.error("⚠️ Riesgo alto de fatalidad")
        else:
            st.success("✅ Riesgo bajo de fatalidad")




st.divider()

st.subheader(" Información del modelo")

st.write("""
Este sistema utiliza un modelo de **Random Forest** entrenado con datos históricos de accidentes de tránsito del municipio de Acacias en el periodo de 2021 - 2024.

El objetivo del modelo es **predecir la severidad del accidente**, clasificando si existe riesgo de fatalidad.

Variables utilizadas:

- Vehículos involucrados
- Mes del accidente
- Fin de semana
- Tipo de accidente

Se aplicó la técnica **SMOTE (Synthetic Minority Oversampling Technique)** para balancear las clases del dataset.
""")




st.subheader(" Importancia de variables en el modelo")

importances = pd.Series(
    model.feature_importances_,
    index=model.feature_names_in_
)

st.bar_chart(importances.sort_values())