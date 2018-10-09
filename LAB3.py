from sklearn.datasets import load_diabetes
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas

diabetes = load_diabetes()
diabetes.keys()

print(diabetes.DESCR)

tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela.head(10)
tabela['Taxa'] = diabetes.target
print(tabela.head(10))
X = tabela[["bmi", "s3"]]

X_t = X[:-20]
X_v = X[-20:]
print(X_t["bmi"])
y_t = tabela["Taxa"][:-20]
y_v = tabela["Taxa"][-20:]

regr = linear_model.LinearRegression()

# treina o modelo
regr.fit(X_t, y_t)

# faz a predição
y_pred = regr.predict(X_v)

# coeficientes a
print('Coeficientes: \n', regr.coef_)
#intercepto b
print('Coeficientes: \n', regr.intercept_)
#y = 5.10*bmi + -0.65*s3 + -1.24

#prediz manualmente os valores com base nos coeficientes encontrados na regressao
y_teste = 814.25596331*X_v["bmi"] - 348.151465*X_v["s3"] + 152.80062545049168

#exibe o valor predito manualmente y_teste, que começa de 486
#exibe o valor real y_t
#exibe o valor predito pela regressão linear

print(y_teste[422], y_t[0],y_pred[0])

#plota todos os valores de validação
plt.scatter(X_v["s3"], y_v,  color='black')
plt.scatter(X_v["s3"], y_pred, color='blue')
plt.legend(["Real", "Predito"])

erro = (((sum(y_v-y_pred)**2))**(1/2))/20
print('erro:',erro)
