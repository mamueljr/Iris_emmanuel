import os
import joblib


def predict1(data, model_name):
    MODELOS_PATH = os.path.join("modelos", f'{model_name}')
    PIPELINE_PATH = os.path.join("modelos", 'Pipeline_Iris.sav')
    model = joblib.load(MODELOS_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    transformed_data = pipeline.transform(data)
    return model.predict(transformed_data)


def predict(data, model_name):
    lr_final = joblib.load(
        'lr_final_model.sav')
    dt_final = joblib.load(
        'dt_final_model.sav')
    svc_final = joblib.load(
        'svm_final_model.sav')
    vc_final = joblib.load(
        'vc_final_model.sav')
    data_transform = joblib.load(
        'Pipeline_Iris.sav')
    transformed_data = data_transform.transform(data)
    return lr_final.predict(transformed_data), svc_final.predict(transformed_data), dt_final.predict(data), vc_final.predict(data)
