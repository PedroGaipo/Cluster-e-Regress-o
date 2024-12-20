import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns #Tem que importar a biblioteca seaborn, usando o pip install seaborn
import matplotlib.pyplot as plt

# 1. Carrega o dataset
data = pd.read_csv('Eminem Dataset.csv')

# 2. Selecionar as variáveis de interesse
# Variável target: explicit (1 para explícita, 0 para não explícita)
# Features: popularity, danceability, energy
features = data[['popularity', 'danceability', 'energy']]
target = data['explicit']

# 3. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# 4. Treinar o modelo de regressão logística
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 5. Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# 6. Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)
report = classification_report(y_test, y_pred)
print("\nRelatório de Classificação:")
print(report)

# Visualizar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Explícita', 'Explícita'],
            yticklabels=['Não Explícita', 'Explícita'])
plt.xlabel('Predição')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()
