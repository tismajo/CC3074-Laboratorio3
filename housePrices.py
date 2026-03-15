import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

"""## 2. Cargar el dataset"""

df = pd.read_csv('train.csv')
df.head()

"""### Información general del dataset"""

df.info()

df.describe()

"""## 3. Análisis Exploratorio de Datos

### Distribución del precio de las viviendas
"""

plt.figure()
sns.histplot(df['SalePrice'], kde=True)
plt.title('Distribución del Precio de las Casas')
plt.show()

"""### Correlación entre variables"""

corr = df.corr(numeric_only=True)

plt.figure()
sns.heatmap(corr)
plt.title('Mapa de correlación')
plt.show()

corr['SalePrice'].sort_values(ascending=False).head(10)

"""## 4. Análisis de Agrupamiento (Clustering)"""

cluster_data = df[['SalePrice','GrLivArea','OverallQual']].dropna()

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data['cluster'] = kmeans.fit_predict(scaled)

sns.scatterplot(x=cluster_data['GrLivArea'], y=cluster_data['SalePrice'], hue=cluster_data['cluster'])
plt.title('Clustering de viviendas')
plt.show()

"""## 5. División de datos en entrenamiento y prueba"""

features = ['GrLivArea','OverallQual','GarageCars','TotalBsmtSF']

X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""## 6. Modelo de Regresión Lineal Simple"""

X_uni = df[['GrLivArea']]
y = df['SalePrice']

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
    X_uni, y, test_size=0.2, random_state=42
)

model_uni = LinearRegression()
model_uni.fit(X_train_u, y_train_u)

pred_uni = model_uni.predict(X_test_u)

plt.scatter(X_test_u, y_test_u)
plt.plot(X_test_u, pred_uni)
plt.title('Regresión Lineal Simple')
plt.show()

print('R2:', r2_score(y_test_u, pred_uni))
print('RMSE:', np.sqrt(mean_squared_error(y_test_u, pred_uni)))

"""## 7. Modelo de Regresión Lineal Múltiple"""

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

pred_multi = model_multi.predict(X_test)

print('R2:', r2_score(y_test, pred_multi))
print('RMSE:', np.sqrt(mean_squared_error(y_test, pred_multi)))

"""## 8. Análisis de Residuos"""

residuals = y_test - pred_multi

sns.scatterplot(x=pred_multi, y=residuals)
plt.axhline(0)
plt.title('Análisis de residuos')
plt.show()

"""## 9. Conclusión

El análisis exploratorio permitió identificar variables importantes que influyen en el precio de las viviendas. Las variables con mayor correlación con el precio incluyen calidad general de la vivienda, área habitable y número de espacios de garaje.

Se construyeron dos modelos:
- Regresión lineal simple
- Regresión lineal múltiple

El modelo múltiple mostró mejor desempeño predictivo, ya que incorpora varias variables relevantes que explican mejor la variabilidad del precio de las casas.

Por lo tanto, el modelo de **regresión lineal múltiple** se considera el más adecuado para la predicción del precio de las viviendas.

## Preprocesamiento de datos

Según el análisis exploratorio, varios valores faltantes representan ausencia de una característica (por ejemplo, casas sin piscina o sin chimenea).  
Por ello:

- Variables categóricas → se reemplazan con `"None"`
- Variables numéricas → se rellenan con la **mediana**

Esto evita perder información al eliminar filas.
"""

def preprocess_data(df):

    categorical = df.select_dtypes(include=["object"]).columns
    df[categorical] = df[categorical].fillna("None")

    numeric = df.select_dtypes(include=[np.number]).columns
    for col in numeric:
        df[col] = df[col].fillna(df[col].median())

    return df

train = preprocess_data(df)

"""## Eliminación de outliers

Durante el análisis exploratorio se detectaron casas extremadamente grandes (`GrLivArea > 4000`) con precios relativamente bajos.

Estas observaciones pueden afectar negativamente modelos lineales, por lo que se eliminan.
"""

train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))]
print("Dataset shape after outlier removal:", train.shape)
