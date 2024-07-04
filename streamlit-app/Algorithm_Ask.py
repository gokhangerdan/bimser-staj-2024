import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Rastgele bir veri seti oluştur
np.random.seed(0)
m2 = 2 * np.random.rand(100, 1)
price = 4 + 3 * m2 + np.random.randn(100, 1)

# Veriyi bir DataFrame'e dönüştür
df = pd.DataFrame(data={'m2': m2.flatten(), 'price': price.flatten()})

# Veriyi X (özellikler) ve y (hedef) olarak ayır
X = df[['m2']].values
y = df['price'].values

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit uygulaması
st.title('Regression Model Comparison')

# Algoritma seçimi
algorithm = st.selectbox('Select an algorithm:', ('Linear Regression', 'Support Vector Machine', 'Nearest Neighbors'))

# Modeli oluştur ve eğit
if algorithm == 'Linear Regression':
    model = LinearRegression()
elif algorithm == 'Support Vector Machine':
    model = SVR(kernel='linear')
elif algorithm == 'Nearest Neighbors':
    model = KNeighborsRegressor(n_neighbors=3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Hata oranlarını hesapla
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'R^2 Score: {r2:.2f}')

# Kullanıcıdan bir metrekare değeri al
m2_input = st.number_input('Enter a m2 value:', min_value=0.0, max_value=2.0, value=1.0, step=0.01)

# Kullanıcı girdisi için fiyat tahmini yap
predicted_price = model.predict(np.array([[m2_input]]))[0]

# Tahmin edilen fiyatı göster
st.write(f'Predicted Price for {m2_input} m2: {predicted_price:.2f}')

# Grafik oluştur
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_train, y_train, color='blue', label='Training data')
ax.scatter(X_test, y_test, color='green', label='Testing data')
ax.plot(X_test, y_pred, color='red', linewidth=2, label='Prediction' if algorithm != 'Nearest Neighbors' else 'Nearest Neighbors')
ax.scatter(m2_input, predicted_price, color='orange', label='User input')
ax.set_title(f'Regression using {algorithm}')
ax.set_xlabel('Metrekare')
ax.set_ylabel('Fiyat')
ax.legend()
ax.grid(True)

# Grafiği Streamlit ile göster
st.pyplot(fig)

# Kullanıcı girdisinin grafikteki konumunu belirten metin
st.write(f'The entered m2 value of {m2_input} is shown in the graph with an orange dot.')
