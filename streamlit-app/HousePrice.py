# HousePrice.py

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import skew

import warnings
warnings.filterwarnings(action="ignore")

#matplotlib.use('TkAgg')  # GG: Buna gerek yok

# Dosya yollarının tanımlanması
train_path = 'data/train.csv'
test_path = 'data/test.csv'

# Veri dosyalarının yüklenmesi
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
house_data = train_data.copy()
data_w = house_data.copy()
data_w.columns = data_w.columns.str.replace(' ', '')

# Streamlit başlığı
st.title("House Price Prediction")

# Model seçimi
model_option = st.selectbox(
    "Choose a model:",
    ("Linear Regression", "K-Nearest Neighbors", "Random Forest")
)

# GG: Burada imputation olup olmayacağını seçtirme. burada yaptığın zaten preprocessing.
# GG: Ayrıca preprocess aşamasından geçirmeden modele veriyi veremezsin
# Veri durumu seçimi
data_option = st.selectbox(
    "Choose data preprocessing option:",
    ("Before Imputation", "After Imputation")
)

# Eksik değerleri doldurmadan önceki veri seti
if data_option == "Before Imputation":
    data = house_data.copy()
    data.columns = data.columns.str.replace(' ', '')
    target = data['SalePrice']
else:
    # Eksik değerleri doldurduktan sonraki veri seti
    all_data = pd.concat([train_data, test_data], sort=False)
    categorical_columns = all_data.select_dtypes(include=['object']).columns
    numerical_columns = all_data.select_dtypes(include=['number']).columns

    for column in categorical_columns:
        all_data[column] = all_data[column].fillna(all_data[column].mode()[0])

    for column in numerical_columns:
        all_data[column] = all_data[column].fillna(all_data[column].median())

    train_test = pd.concat([train_data, test_data], sort=False)
    train_test["SqFtPerRoom"] = train_test["GrLivArea"] / (train_test["TotRmsAbvGrd"] +
                                                           train_test["FullBath"] +
                                                           train_test["HalfBath"] +
                                                           train_test["KitchenAbvGr"])
    train_test['Total_Home_Quality'] = train_test['OverallQual'] + train_test['OverallCond']
    train_test['Total_Bathrooms'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) +
                                     train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))
    train_test["HighQualSF"] = train_test["1stFlrSF"] + train_test["2ndFlrSF"]

    train_test['MSSubClass'] = train_test['MSSubClass'].astype(str)
    train_test['YrSold'] = train_test['YrSold'].astype(str)
    train_test['MoSold'] = train_test['MoSold'].astype(str)

    train_test_dummy = pd.get_dummies(train_test)

    numeric_features = train_test_dummy.select_dtypes(include=[np.number]).columns
    skewed_features = train_test_dummy[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_features[skewed_features > 0.5]
    skew_index = high_skew.index

    for i in skew_index:
        train_test_dummy[i] = np.log1p(train_test_dummy[i])

    data = train_test_dummy.copy()
    target = np.log1p(train_data['SalePrice'])

# Model seçimine göre model oluşturma
if model_option == "Linear Regression":
    model = LinearRegression()
elif model_option == "K-Nearest Neighbors":
    model = KNeighborsRegressor()
elif model_option == "Random Forest":
    model = RandomForestRegressor()

# Model eğitimi ve sonuçların gösterimi
if st.button("Train Model"):
    # Veri ve hedef ayrımı
    X = data.drop(columns=['SalePrice'], errors='ignore')
    y = target

    # Eğitim ve test setlerine bölme
    # GG: Aşağıdaki X, y aynı uzunlukta olmalı
    st.write(len(X))
    st.write(len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # GG: Buradan modele vereceğin veriyi kontrol et
    st.write(X_train)
    st.write(y_train)

    # Model eğitimi
    model.fit(X_train, y_train)

    # Tahminler
    predictions = model.predict(X_test)

    # Hata hesaplama
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Sonuçların gösterimi
    st.write(f"Model: {model_option}")
    st.write(f"Data Preprocessing: {data_option}")
    st.write(f"RMSE: {rmse}")

    # Grafiklerin gösterimi
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.2)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{model_option} Predictions')
    st.pyplot(fig)
