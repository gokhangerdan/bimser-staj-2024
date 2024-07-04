import pandas as pd
import numpy as np

# Rastgele bir veri seti oluştur
np.random.seed(0)
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

df.to_csv('example_data.csv', index=False)

import pandas as pd
import numpy as np

# Rastgele bir veri seti oluştur
np.random.seed(0)
df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})

df.to_csv('example_data.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Veriyi yükle
df = pd.read_csv('example_data.csv')

# Özellikler ve hedef değişkeni ayır
X = df[['feature1', 'feature2']]
y = df['target']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Doğruluk skorunu hesapla
accuracy = accuracy_score(y_test, y_pred)

print(f'Model Accuracy: {accuracy:.2f}')
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Başlık
st.title('Random Forest Classifier')

# Veriyi yükle
df = pd.read_csv('example_data.csv')

# Özellikler ve hedef değişkeni ayır
X = df[['feature1', 'feature2']]
y = df['target']

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Doğruluk skorunu hesapla
accuracy = accuracy_score(y_test, y_pred)

# Doğruluk skorunu göster
st.write(f'Model Accuracy: {accuracy:.2f}')

# Özellik önemlerini görselleştir
feature_importances = model.feature_importances_
features = X.columns

fig, ax = plt.subplots()
ax.barh(features, feature_importances)
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Feature Importances')
st.pyplot(fig)



