import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Rastgele bir veri seti oluştur
np.random.seed(0)
m2 = 2 * np.random.rand(100, 1)
price = 4 + 3 * m2 + np.random.randn(100, 1)

# Veriyi bir DataFrame'e dönüştür
df = pd.DataFrame(data={'m2': m2.flatten(), 'price': price.flatten()})

# Veriyi X (özellikler) ve y (hedef) olarak ayır
X = df[['m2']]
y = df['price']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer regresyon modelini oluştur
model = LinearRegression()

# Modeli eğit
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# Hata oranlarını hesapla
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Eğitim setini görselleştir
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prediction')

plt.title('Linear Regression')
plt.xlabel('Metrekare')
plt.ylabel('Fiyat')
plt.legend()
plt.show()

