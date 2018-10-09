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
