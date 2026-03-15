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

"""## Modelo de regresión utilizando Log(SalePrice)"""

train.loc[:, "LogSalePrice"] = np.log1p(train["SalePrice"])
y_log = train["LogSalePrice"]

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

model_log = LinearRegression()
model_log.fit(X_train_log, y_train_log)

pred_log = model_log.predict(X_test_log)

print("Log Regression Model")
print("R2:", r2_score(y_test_log, pred_log))
print("RMSE:", np.sqrt(mean_squared_error(y_test_log, pred_log)))

"""## Multicolinealidad usando VIF (Variance Inflation Factor)"""

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)

"""Interpretación del VIF:

- **VIF = 1** → sin correlación
- **VIF entre 1 y 5** → correlación moderada
- **VIF > 5** → posible multicolinealidad

## Diagnóstico de residuos
"""

residuals = y_test - pred_multi

plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Distribución de residuos")
plt.show()

plt.figure()
sns.scatterplot(x=pred_multi, y=residuals)
plt.axhline(0)
plt.title("Residuos vs Predicciones")
plt.show()

"""## Evaluación de overfitting"""

train_pred = model_multi.predict(X_train)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, pred_multi)

print("Train R2:", train_r2)
print("Test R2:", test_r2)

"""Si el **R² de entrenamiento es mucho mayor que el de prueba**, el modelo puede estar sobreajustado (overfitting).

## Discusión final

El análisis exploratorio permitió identificar los factores más relevantes en el precio de las viviendas. Variables relacionadas con **calidad de construcción, tamaño habitable y características estructurales** muestran la mayor influencia.

Se evaluaron tres modelos:

1. Regresión lineal simple
2. Regresión lineal múltiple
3. Regresión con transformación logarítmica

El modelo múltiple y el modelo log-transformado tienden a mostrar mejor desempeño que la regresión simple, ya que incorporan más información sobre la vivienda.

Además, el análisis de multicolinealidad mediante VIF permitió verificar que las variables seleccionadas no presentan dependencia excesiva entre sí.

Finalmente, el modelo múltiple se considera el más apropiado para la predicción del precio de viviendas en este dataset.

El análisis realizado sobre el Ames Housing Dataset permitió identificar los factores más importantes que influyen en el precio de venta de una vivienda. A través del análisis exploratorio se observó que la variable objetivo SalePrice presenta una distribución con fuerte asimetría positiva, lo que indica que existen pocas viviendas con precios extremadamente altos. Este comportamiento es común en datasets inmobiliarios, donde las propiedades de lujo generan valores atípicos que pueden afectar modelos estadísticos tradicionales.

El análisis de correlación permitió identificar las variables con mayor relación con el precio de la vivienda. Entre las más relevantes se encuentran OverallQual, GrLivArea, GarageCars, GarageArea y TotalBsmtSF. Estas variables representan principalmente la calidad de la construcción, el tamaño de la propiedad y la disponibilidad de espacios adicionales, factores que tradicionalmente influyen en el valor de una vivienda dentro del mercado inmobiliario.

En particular, la variable OverallQual mostró la correlación más alta con el precio de venta, lo que sugiere que la calidad de los materiales y acabados tiene un impacto significativo en la valoración de la propiedad. De forma similar, GrLivArea mostró una relación lineal positiva clara, indicando que las viviendas con mayor área habitable tienden a presentar precios más altos. Estas observaciones coinciden con principios económicos básicos del mercado inmobiliario, donde el tamaño y la calidad de la construcción son determinantes clave del valor de una propiedad.

Durante el análisis también se identificaron outliers, principalmente viviendas con áreas habitables extremadamente grandes pero con precios relativamente bajos. Estas observaciones pueden distorsionar los modelos de regresión lineal, por lo que se aplicó una estrategia de eliminación de estos casos para mejorar la estabilidad del modelo.

Posteriormente se construyeron diferentes modelos de regresión con el objetivo de predecir el precio de las viviendas. En primer lugar se desarrolló un modelo de regresión lineal simple utilizando la variable GrLivArea como predictor. Este modelo permitió observar la relación directa entre el tamaño de la vivienda y su precio, aunque su capacidad predictiva es limitada al considerar únicamente una variable.

Para mejorar la predicción se construyó un modelo de regresión lineal múltiple, incorporando varias de las variables con mayor correlación con SalePrice. Este modelo mostró un mejor desempeño, ya que logra capturar múltiples factores estructurales de la vivienda que influyen simultáneamente en su precio. El análisis de residuos mostró que el modelo logra ajustarse razonablemente bien a los datos, aunque aún existe cierta variabilidad inherente al mercado inmobiliario.

Adicionalmente se evaluó la presencia de multicolinealidad entre las variables predictoras mediante análisis de correlación y el cálculo del factor de inflación de la varianza (VIF). Los resultados indicaron que las variables seleccionadas presentan niveles aceptables de correlación, por lo que pueden utilizarse conjuntamente dentro del modelo sin generar problemas significativos de redundancia en la información.

Finalmente, se evaluó el desempeño de los modelos utilizando el conjunto de prueba, empleando métricas como R² y RMSE para medir la calidad de la predicción. Los resultados muestran que el modelo de regresión múltiple presenta un mejor desempeño que el modelo univariado, ya que logra explicar una mayor proporción de la variabilidad en los precios de las viviendas.

En conclusión, el análisis realizado demuestra que el precio de una vivienda depende de múltiples factores relacionados principalmente con la calidad de la construcción, el tamaño de la propiedad y sus características estructurales. Entre los modelos evaluados, la regresión lineal múltiple se presenta como el enfoque más adecuado para predecir el precio de las viviendas en este conjunto de datos, ya que permite integrar diversas variables relevantes y obtener estimaciones más precisas.
"""