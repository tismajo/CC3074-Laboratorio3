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

"""## Ingeniería de características

Basado en el análisis de correlación, se seleccionaron variables con alta relación con el precio:

- OverallQual
- GrLivArea
- GarageCars
- TotalBsmtSF
- YearBuilt

Estas variables representan calidad de construcción, tamaño de la vivienda y características estructurales importantes.
"""

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt"
]

X = train[features]
y = train["SalePrice"]

"""## División de datos en entrenamiento y prueba"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

"""## Modelo de regresión lineal simple

Se selecciona la variable **GrLivArea**, ya que mostró una relación lineal fuerte con el precio.
"""

from sklearn.linear_model import LinearRegression

X_train_uni = X_train[["GrLivArea"]]
X_test_uni = X_test[["GrLivArea"]]

model_uni = LinearRegression()
model_uni.fit(X_train_uni, y_train)

pred_uni = model_uni.predict(X_test_uni)

plt.scatter(X_test_uni, y_test)
plt.plot(X_test_uni, pred_uni)
plt.title("Linear Regression: GrLivArea vs SalePrice")
plt.show()

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

print("Simple Regression")
print("R2:", r2_score(y_test, pred_uni))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_uni)))

"""## Modelo de regresión lineal múltiple"""

model_multi = LinearRegression()

model_multi.fit(X_train, y_train)

pred_multi = model_multi.predict(X_test)

print("Multiple Regression")
print("R2:", r2_score(y_test, pred_multi))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred_multi)))

"""## Análisis de multicolinealidad"""

corr = X.corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

"""## Análisis de residuos"""

residuals = y_test - pred_multi

plt.scatter(pred_multi, residuals)
plt.axhline(0)
plt.title("Residual Analysis")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

"""## Conclusión

El análisis exploratorio permitió identificar las variables más influyentes en el precio de las viviendas, especialmente la calidad de la construcción (`OverallQual`), el área habitable (`GrLivArea`) y el tamaño del garaje.

Se construyeron dos modelos:

- Regresión lineal simple
- Regresión lineal múltiple

El modelo de regresión múltiple mostró mejor desempeño predictivo, ya que combina múltiples variables que explican mejor la variabilidad del precio.

Por lo tanto, el modelo múltiple se considera el más adecuado para predecir el precio de las viviendas en este dataset.

## Transformación logarítmica de la variable objetivo

Durante el análisis exploratorio se observó que `SalePrice` presenta **asimetría positiva**.  
Una práctica común en regresión es aplicar una transformación logarítmica para:

- Reducir el efecto de outliers
- Mejorar la normalidad de los residuos
- Estabilizar la varianza
"""

train["LogSalePrice"] = np.log1p(train["SalePrice"])

plt.figure()
sns.histplot(train["LogSalePrice"], kde=True)
plt.title("Distribución Log(SalePrice)")
plt.show()