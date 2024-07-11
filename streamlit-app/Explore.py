import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns #Seaborn, istatistiksel veri görselleştirmeleri için kullanılan bir kütüphanedir.
import scipy.stats as stats #SciPy kütüphanesinin istatistiksel fonksiyonlarını kullanmak için import edilmiştir.
import statsmodels.api as sm #Statsmodels, istatistiksel modellerin tahmini ve istatistiksel testler için kullanılır.
import matplotlib.pyplot as plt #Matplotlib, 2D grafikler ve görselleştirmeler oluşturmak için kullanılır.
import matplotlib

from sklearn.neighbors import KNeighborsRegressor #Sklearn (scikit-learn) kütüphanesinden K-Nearest Neighbors Regressor modelini içe aktarıyoruz.
from scipy.stats import norm #normal dağılımı temsil eden bir fonksiyondur
from scipy.stats import skew, norm # Çarpıklık, bir dağılımın simetrisizliğini ölçer. Pozitif çarpıklık, sağa çarpık bir dağılımı, negatif çarpıklık ise sola çarpık bir dağılımı gösterir
from scipy.stats import kurtosis, shapiro

import warnings
warnings.filterwarnings(action="ignore") #Uyarıları görmezden gelmek için kullanılır. Bu, daha temiz bir çıktı sağlar ve önemli olmayan uyarıları gizler.

matplotlib.use('TkAgg')

# Dosya yollarının tanımlanması
train_path = 'data/train.csv'
test_path = 'data/test.csv'

# Veri dosyalarının yüklenmesi
house_data = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Veri setinin bir kopyasını oluşturma ve sütun isimlerindeki boşlukları kaldırma
data_w = house_data.copy()
data_w.columns = data_w.columns.str.replace(' ', '')

# Veri setinin genel yapısının incelenmesi
#print(data_w.info())

'''
(mu, sigma) = norm.fit(data_w['SalePrice'])

plt.figure(figsize = (12,6))
sns.distplot(data_w['SalePrice'], kde = True, hist=True, fit = norm)
plt.title('SalePrice distribution vs Normal Distribution', fontsize = 13)
plt.xlabel("House's sale Price in $", fontsize = 12)
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.show()
'''


'''
# Skewness ve Kurtosis hesaplama
saleprice_skewness = skew(data_w['SalePrice'])
saleprice_kurtosis = kurtosis(data_w['SalePrice'])

print(f"Skewness: {saleprice_skewness}")
print(f"Kurtosis: {saleprice_kurtosis}")

# Shapiro-Wilk testi
shapiro_test = shapiro(data_w['SalePrice'])
print(f"Shapiro-Wilk Test: W={shapiro_test[0]}, p-value={shapiro_test[1]}")
'''

'''
# Sadece sayısal sütunları seçme
numeric_data_w = data_w.select_dtypes(include=[np.number])

# Korelasyon matrisi oluşturma
correlation_matrix = numeric_data_w.corr()

# Korelasyon matrisinin görüntülenmesi
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 7}, linewidths=0.5)
plt.xticks(fontsize=10, rotation=90)
plt.yticks(fontsize=10)
plt.title('Correlation Matrix')
plt.show()

# Hedef değişken ile en yüksek korelasyona sahip özellikleri bulma
correlation_with_target = correlation_matrix['SalePrice'].sort_values(ascending=False)
print("Features with highest correlation with SalePrice:")
print(correlation_with_target.head(10))  # En yüksek korelasyona sahip ilk 10 özelliği göster
'''

'''
# OverallQual ve SalePrice arasındaki ilişkinin incelenmesi
figure, ax = plt.subplots(1, 3, figsize=(20, 8))

# Strip Plot
sns.stripplot(data=data_w, x='OverallQual', y='SalePrice', ax=ax[0])
ax[0].set_title('Strip Plot of OverallQual vs SalePrice')

# Violin Plot
sns.violinplot(data=data_w, x='OverallQual', y='SalePrice', ax=ax[1])
ax[1].set_title('Violin Plot of OverallQual vs SalePrice')

# Box Plot
sns.boxplot(data=data_w, x='OverallQual', y='SalePrice', ax=ax[2])
ax[2].set_title('Box Plot of OverallQual vs SalePrice')

#  Strip plot, her kalite seviyesindeki bireysel veri noktalarını gösterir; violin plot, bu seviyelerdeki veri yoğunluğunu gösterir; box plot ise verilerin çeyrekler arası dağılımını ve medyanını özetler.
plt.show()
'''

'''
# GrLivArea vs SalePrice [corr = 0.71]

Pearson_GrLiv = 0.71
plt.figure(figsize = (12,6))
sns.regplot(data=data_w, x = 'GrLivArea', y='SalePrice', scatter_kws={'alpha':0.2})
plt.title('GrLivArea vs SalePrice', fontsize = 12)
plt.legend(['$Pearson=$ {:.2f}'.format(Pearson_GrLiv)], loc = 'best')
plt.show()
'''
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Veri setlerinin birleştirilmesi
all_data = pd.concat([train_data, test_data], sort=False)

# Eksik değerlerin kontrol edilmesi
missing_values = all_data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
#print("Missing values in the dataset:")
#print(missing_values)

'''
# Eksik değerlerin yüzdesinin hesaplanması
nan_percentage = (all_data.isnull().sum() / len(all_data)) * 100
nan_percentage = nan_percentage[nan_percentage > 0].sort_values(ascending=False)
nan_df = pd.DataFrame({'feat': nan_percentage.index, 'Perc(%)': nan_percentage.values})

# Eksik değerlerin görselleştirilmesi
plt.figure(figsize=(15, 5))
sns.barplot(x='feat', y='Perc(%)', data=nan_df, palette='viridis')
plt.xticks(rotation=45)
plt.title('Features containing NaN')
plt.xlabel('Features')
plt.ylabel('% of Missing Data')
plt.show()
'''

# Kategorik ve sayısal sütunların belirlenmesi
categorical_columns = all_data.select_dtypes(include=['object']).columns
numerical_columns = all_data.select_dtypes(include=['number']).columns

# Kategorik değişkenlerin eksik değerlerini doldurma
for column in categorical_columns:
    all_data[column] = all_data[column].fillna(all_data[column].mode()[0])

# Sayısal değişkenlerin eksik değerlerini doldurma
for column in numerical_columns:
    all_data[column] = all_data[column].fillna(all_data[column].median())

# Eksik değerlerin tekrar kontrol edilmesi
missing_values = all_data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
#print("Remaining missing values in the dataset:")
#print(missing_values)

train_test = pd.concat([train_data, test_data], sort=False)

# Yeni özelliklerin oluşturulması
train_test["SqFtPerRoom"] = train_test["GrLivArea"] / (train_test["TotRmsAbvGrd"] +
                                                       train_test["FullBath"] +
                                                       train_test["HalfBath"] +
                                                       train_test["KitchenAbvGr"])

train_test['Total_Home_Quality'] = train_test['OverallQual'] + train_test['OverallCond']

train_test['Total_Bathrooms'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) +
                                 train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))

train_test["HighQualSF"] = train_test["1stFlrSF"] + train_test["2ndFlrSF"]

# Sayısal olarak saklanan kategorik değişkenleri string'e dönüştürme
train_test['MSSubClass'] = train_test['MSSubClass'].astype(str)
train_test['YrSold'] = train_test['YrSold'].astype(str)
train_test['MoSold'] = train_test['MoSold'].astype(str)

# Kategorik değişkenlerden dummy değişkenlerin oluşturulması
train_test_dummy = pd.get_dummies(train_test)

# Sayısal özellikleri alma ve çarpık özelliklerin belirlenmesi
numeric_features = train_test_dummy.select_dtypes(include=[np.number]).columns
skewed_features = train_test_dummy[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_features[skewed_features > 0.5]
skew_index = high_skew.index

# Çarpık özellikleri log dönüşümü ile normalleştirme
for i in skew_index:
    train_test_dummy[i] = np.log1p(train_test_dummy[i])

# İşlenmiş veri setini inceleme
#print(train_test_dummy.head())


target = train_data['SalePrice']

fig, ax = plt.subplots(1,2, figsize= (15,5))
fig.suptitle(" qq-plot & distribution SalePrice ", fontsize= 15)

# QQ-plot
sm.qqplot(target, line="s", ax=ax[0])
ax[0].set_title('QQ-plot of SalePrice')
sns.distplot(target, kde = True, hist=True, fit = norm, ax = ax[1])

target_log = np.log1p(target)

fig, ax = plt.subplots(1,2, figsize= (15,5))
fig.suptitle("qq-plot & distribution SalePrice ", fontsize= 15)

sm.qqplot(target_log, line="s", ax=ax[0])
ax[0].set_title('QQ-plot of Log-transformed SalePrice') 

sns.distplot(target_log, kde = True, hist=True, fit = norm, ax = ax[1])
plt.show()
