import streamlit as st
import uvicorn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import requests
import json

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from scipy.stats import skew
from fastapi import FastAPI

import warnings
warnings.filterwarnings(action="ignore")

# Dosya yollarının tanımlanması
train_path = 'Data/train.csv'
test_path = 'Data/test.csv'

# Veri dosyalarının yüklenmesi
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
house_data = train_data.copy()
data_w = house_data.copy()
data_w.columns = data_w.columns.str.replace(' ', '')

# FastAPI uygulaması oluşturma
app = FastAPI()

# Veri işleme fonksiyonu
def preprocess_data(data_option):
    if data_option == "Before Skew":
        data = house_data.copy()
        data.columns = data.columns.str.replace(' ', '')
        target = data['SalePrice']
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        numerical_columns = data.select_dtypes(include=['number']).columns

        categorical_columns = [col for col in categorical_columns if col in data.columns]
        numerical_columns = [col for col in numerical_columns if col in data.columns]

        if categorical_columns:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            data[categorical_columns] = imputer_cat.fit_transform(data[categorical_columns])

        if numerical_columns:
            imputer_num = SimpleImputer(strategy='median')
            data[numerical_columns] = imputer_num.fit_transform(data[numerical_columns])

        data = pd.get_dummies(data)
    
    else:
        all_data = pd.concat([train_data, test_data], sort=False)
        categorical_columns = all_data.select_dtypes(include=['object']).columns
        numerical_columns = all_data.select_dtypes(include=['number']).columns

        for column in categorical_columns:
            if column in all_data.columns:
                all_data[column] = all_data[column].fillna(all_data[column].mode()[0])

        for column in numerical_columns:
            if column in all_data.columns:
                all_data[column] = all_data[column].fillna(all_data[column].median())

        all_data["SqFtPerRoom"] = all_data["GrLivArea"] / (all_data["TotRmsAbvGrd"] +
                                                           all_data["FullBath"] +
                                                           all_data["HalfBath"] +
                                                           all_data["KitchenAbvGr"])
        all_data['Total_Home_Quality'] = all_data['OverallQual'] + all_data['OverallCond']
        all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                                       all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))
        all_data["HighQualSF"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"]

        all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
        all_data['YrSold'] = all_data['YrSold'].astype(str)
        all_data['MoSold'] = all_data['MoSold'].astype(str)

        all_data_dummy = pd.get_dummies(all_data)

        numeric_features = all_data_dummy.select_dtypes(include=[np.number]).columns
        skewed_features = all_data_dummy[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
        high_skew = skewed_features[skewed_features > 0.5]
        skew_index = high_skew.index

        for i in skew_index:
            all_data_dummy[i] = np.log1p(all_data_dummy[i])

        train_data_processed = all_data_dummy[:len(train_data)]
        test_data_processed = all_data_dummy[len(train_data):]

        data = train_data_processed.copy()
        target = np.log1p(train_data['SalePrice'])

        categorical_columns = [col for col in categorical_columns if col in data.columns]
        if categorical_columns:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            data[categorical_columns] = imputer_cat.fit_transform(data[categorical_columns])

        numerical_columns = [col for col in numerical_columns if col in data.columns]
        if numerical_columns:
            imputer_num = SimpleImputer(strategy='median')
            data[numerical_columns] = imputer_num.fit_transform(data[numerical_columns])

    return data, target

# Model eğitimi ve kaydetme fonksiyonu
def train_and_save_model(model_option, data, target):
    if model_option == "Linear Regression":
        model = LinearRegression()
    elif model_option == "K-Nearest Neighbors":
        model = KNeighborsRegressor()
    elif model_option == "Random Forest":
        model = RandomForestRegressor()

    X = data.drop(columns=['SalePrice'], errors='ignore')
    y = target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Modeli pickle ile kaydetme
    model_filename = f"model_{model_option.replace(' ', '_').lower()}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    return model, rmse, y_test, predictions, model_filename

# FastAPI endpoint
@app.post("/train/")
async def train_api(model_option: str, data_option: str):
    data, target = preprocess_data(data_option)
    model, rmse, y_test, predictions, model_filename = train_and_save_model(model_option, data, target)
    return {"model": model_option, "data_preprocessing": data_option, "rmse": rmse, "y_test": y_test.tolist(), "predictions": predictions.tolist(), "model_filename": model_filename}

# Streamlit başlığı
st.title("House Price Prediction")

# Model seçimi
model_option = st.selectbox(
    "Choose a model:",
    ("Linear Regression", "K-Nearest Neighbors", "Random Forest")
)

# Veri durumu seçimi
data_option = st.selectbox(
    "Choose data preprocessing option:",
    ("Before Skew", "After Skew")
)

# Streamlit arayüzü
if st.button("Train Model"):
    # FastAPI endpoint'ine veri gönderme
    url = "http://localhost:8002/train/"
    payload = {
        "model_option": model_option,
        "data_option": data_option
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        result = response.json()
        rmse = result["rmse"]
        model_filename = result["model_filename"]
        y_test = result["y_test"]
        predictions = result["predictions"]
        
        st.write(f"Model: {model_option}")
        st.write(f"Data Preprocessing: {data_option}")
        st.write(f"RMSE: {rmse}")
        st.write(f"Model saved as: {model_filename}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.2)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{model_option} Predictions')
        st.pyplot(fig)

# FastAPI uygulamasını çalıştırma
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

'''
Before Imputation: Temel eksik değer doldurma ve kategorik değişkenlerden dummy değişkenler oluşturma işlemleri yapılır.

After Imputation: Eksik değer doldurma işlemlerine ek olarak, yeni özellikler oluşturulur, veri dönüşüm işlemleri yapılır ve sayısal özelliklerin çarpıklığı düzeltilir.
'''