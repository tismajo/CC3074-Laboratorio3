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

