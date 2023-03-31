import joblib
import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import time

def predict(data, model_name):
    MODELOS_PATH = os.path.join("modelos",f'{model_name}')
    PIPELINE_PATH = os.path.join("modelos", "Pipeline_Iris.sav")
    model = joblib.load(MODELOS_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    transformed_data = pipeline.transform(data)
    return model.predict(transformed_data)

#import os
# Despliegue de la app
st.title('Predictor para tipo de flor Iris')
#st.write(housing.head())
st.sidebar.header('Datos de Entrada:')
sepal_length = st.sidebar.slider('Sepal length (cm):', 1.0, 8.0, 4.9)
sepal_width = st.sidebar.slider('Sepal width (cm):', 0.1, 5.0, 3.0)
petal_length = st.sidebar.slider('Petal length (cm):', 1.0, 8.0, 1.4)
petal_width = st.sidebar.slider('Petal width (cm):', 0.1, 5.0, 0.2)

st.header('Selecciona un modelo')
model = st.selectbox('Modelo', [
                     'Linear Regressor', 'Random Forest Regressor', 'Decision Tree Regressor'])