
# Classificação de Modelos de Árvore de Decisão: Critérios para Escolha do Melhor Modelo

Este repositório implementa um algoritmo de **Árvore de Decisão** para classificação utilizando os critérios de avaliação de desempenho: **gini** e **entropy**. A escolha do melhor modelo é baseada em diferentes métricas, como **média**, **mediana**, **desvio padrão** da acurácia e **análise da matriz de confusão**.

## Visão Geral do Código

Este código treina modelos de árvore de decisão para uma base de dados de classificação e utiliza dois critérios de impureza: `gini` e `entropy`. A ideia principal é avaliar esses modelos em diferentes profundidades de árvore (`max_depth`) e selecionar o melhor modelo com base em métricas de desempenho. O código também inclui a comparação com um **DummyClassifier** para fornecer uma linha de base.

### Principais Etapas

1. **Carregamento e Divisão dos Dados**
   - Os dados são carregados e divididos em conjuntos de treino e teste. Neste exemplo, utiliza-se a base de dados **Iris**.
   ```python
   X, y = carregar_dados()
   X_train, X_test, y_train, y_test = dividir_dados(X, y)
   ```

2. **Treinamento e Avaliação dos Modelos com Critério 'gini' e 'entropy'**
   - Para ambos os critérios, os modelos são treinados com diferentes profundidades (`max_depth` variando de 1 a 10). A acurácia é avaliada em cada profundidade e armazenada para análise posterior.
   ```python
   for max_depth in profundidades:
       modelo_gini = treinar_arvore_decisao(X_train, y_train, criterio='gini', max_depth=max_depth)
       acuracia_gini, _, _ = avaliar_modelo(modelo_gini, X_test, y_test)
   ```

3. **Análise Estatística dos Resultados**
   - As acurácias são analisadas usando **média**, **mediana** e **desvio padrão**, o que permite avaliar a consistência e variabilidade do desempenho dos modelos ao longo das diferentes profundidades.
   ```python
   calcular_estatisticas_acuracia(acuracias_gini)
   calcular_estatisticas_acuracia(acuracias_entropy)
   ```

4. **Seleção do Melhor Modelo**
   - O melhor modelo é escolhido com base em dois critérios:
     - A **acurácia** deve estar acima da **média** e **mediana** da acurácia de todas as profundidades testadas, sendo a primeira a mais próxima de ambas.
     - A **matriz de confusão** é usada para identificar o modelo com menos erros nas extremidades e mais sucesso na diagonal principal (ou seja, com mais acertos nas classificações corretas).
   - Todos os resultados de matrizes de confusão são exibidos no console para facilitar a comparação visual.
   ```python
   print(f"\nMelhor critério: {melhor_criterio}, com Acurácia={melhor_acuracia:.2f} e max_depth={melhor_max_depth}")
   ```

5. **Visualização e Comparação Final**
   - O melhor modelo é visualizado, e sua matriz de confusão é plotada para análise.
   - A comparação final é feita com o **DummyClassifier**, que atua como uma baseline de referência para verificar se o modelo treinado realmente apresenta um desempenho superior.
   ```python
   dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
   dummy_clf.fit(X_train, y_train)
   y_pred_dummy = dummy_clf.predict(X_test)
   ```

## Critérios para Escolha do Melhor Modelo

### 1. Acurácia
A acurácia é a métrica inicial para avaliar o desempenho dos modelos. Ela indica a proporção de previsões corretas em relação ao total de previsões feitas.

A **acurácia média** e a **mediana** são calculadas para cada critério (gini e entropy) ao longo das diferentes profundidades de árvore. O modelo com acurácia mais próxima da média e mediana (e acima delas) é preferido, pois isso indica um desempenho consistente.

Além disso, é essencial observar o **desvio padrão** da acurácia ao longo das profundidades. Um desvio padrão menor sugere que o desempenho do modelo é menos variável e mais previsível.

### 2. Matriz de Confusão
A **matriz de confusão** é uma ferramenta fundamental para avaliar o desempenho do modelo de forma mais detalhada, além da simples acurácia. Ela permite observar onde ocorrem os erros de classificação.

Os principais pontos de interesse são:
- **Diagonal Principal**: representa o número de acertos para cada classe. O modelo ideal deve ter a maior concentração de valores nessa diagonal.
- **Erros nas Extremidades**: são erros mais graves de classificação. Modelos que minimizam esses erros são preferidos.

### 3. Escolha do Melhor Critério
Após avaliar tanto a acurácia quanto a matriz de confusão, o critério de impureza (gini ou entropy) com melhor desempenho é escolhido.

### 4. Comparação com DummyClassifier
Por fim, o desempenho do melhor modelo é comparado com um **DummyClassifier**. Esse classificador atua de forma ingênua, sempre escolhendo a classe mais frequente. Isso fornece uma linha de base para verificar se o modelo de árvore de decisão está realmente agregando valor ao problema.

## Considerações Finais

Este código visa fornecer uma abordagem equilibrada para a escolha do melhor modelo de árvore de decisão, utilizando tanto métricas estatísticas de desempenho quanto a análise visual da matriz de confusão. Esses critérios garantem que o modelo escolhido tenha um bom desempenho geral e evite erros graves, especialmente em problemas de classificação com várias classes.
