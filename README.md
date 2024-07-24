### Classificação de Comportamento de Clientes em Site

Este repositório contém um exemplo de classificação binária para prever se um cliente comprou ou não com base em seu comportamento de navegação em um site. Utilizamos diferentes abordagens de modelagem para alcançar isso, detalhadas abaixo.

### Arquivos

- **tracking.csv**: Arquivo CSV contendo os dados de comportamento dos clientes no site, incluindo variáveis como "principal", "como_funciona", "contato" e "comprou".(modelo ja utilizado em outro codigo)

### Pré-processamento de Dados

Os dados foram carregados diretamente de um arquivo CSV hospedado online e renomeamos as colunas para facilitar a compreensão:

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/paolasouza/data_mining_and_big_data/ec70f701a784820fa6ca326c0d51d8740028da03/tracking.csv'
dados = pd.read_csv(url)

# Renomear colunas
mapa = {
    "home": "principal",
    "how_it_works": "como_funciona",
    "contact": "contato",
    "bought": "comprou"
}
dados = dados.rename(columns=mapa)
```

### Modelagem e Avaliação

#### 1. Naive Bayes Gaussian

Utilizamos o Naive Bayes Gaussiano para criar um modelo de classificação:

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Separar features e target
x = dados[["principal", "como_funciona", "contato"]]
y = dados["comprou"]

# Dividir dados em treino e teste
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=0)

# Criar modelo
modelo_NB = GaussianNB()
modelo_NB.fit(train_x, train_y)

# Avaliar acurácia
previsoes_NB = modelo_NB.predict(test_x)
acuracia_NB = accuracy_score(test_y, previsoes_NB)
print(f"Acurácia do Naive Bayes Gaussian: {acuracia_NB:.2f}")
```

#### 2. Linear SVC

Também aplicamos o Linear SVC como modelo de classificação:

```python
from sklearn.svm import LinearSVC

# Criar modelo
modelo_SVC = LinearSVC()
modelo_SVC.fit(train_x, train_y)

# Avaliar acurácia
previsoes_SVC = modelo_SVC.predict(test_x)
acuracia_SVC = accuracy_score(test_y, previsoes_SVC)
print(f"Acurácia do Linear SVC: {acuracia_SVC:.2f}")
```

#### 3. Dummy Classifier

Para fins de comparação, utilizamos o Dummy Classifier como baseline:

```python
from sklearn.dummy import DummyClassifier

# Criar modelo dummy
dummy_stratified = DummyClassifier(strategy="stratified")
dummy_stratified.fit(train_x, train_y)

# Avaliar acurácia
acuracia_dummy = dummy_stratified.score(test_x, test_y)
print(f"Acurácia do Dummy Classifier (stratified): {acuracia_dummy:.2f}")
```

### Conclusão

Este projeto demonstra diferentes técnicas de classificação binária aplicadas ao contexto de comportamento de clientes em um site. Cada modelo foi avaliado quanto à sua acurácia na previsão de compra com base nas interações dos clientes.
