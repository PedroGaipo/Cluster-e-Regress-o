import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
data = pd.read_csv('Eminem Dataset.csv')

# Selecionar a coluna relevante para clusterização
features = data[['popularity']]

# Aplicar KMeans com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(features)

# Visualizar os clusters em um gráfico de dispersão
plt.figure(figsize=(12, 8))
plt.scatter(data['track_name'], data['popularity'], c=data['cluster'], cmap='viridis', s=100)
plt.xlabel('Nome da Faixa', fontsize=12)
plt.ylabel('Popularidade', fontsize=12)
plt.title('Clusterização de Músicas por Popularidade', fontsize=14)
plt.xticks(rotation=90)
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.savefig('grafico.png') #COMANDO QUE FAZ COM QUE UMA CÓPIA DO GRÁFICO VÁ PARA A PASTA DO ARQUIVO.
