from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# carrega os dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# cria o objeto do modelo de regressão logística
model = LogisticRegression()

# treina o modelo com os dados de treinamento
model.fit(X_train, y_train)

# faz as previsões com os dados de teste
y_pred = model.predict(X_test)

# calcula a acurácia das previsões
accuracy = accuracy_score(y_test, y_pred)

# imprime a acurácia do modelo
print("Acurácia:", accuracy)
