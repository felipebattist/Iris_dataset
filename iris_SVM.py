from sklearn import datasets		
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

iris = datasets.load_iris()

x = iris.data
y = iris.target
C = 1.0

svc = svm.SVC(kernel='linear', C=C).fit(x, y)


testes = open('iris_dataset.txt', 'r')
testes = testes.readlines()
for linha in testes:
    linha = linha.rstrip('\n')
    linha = linha.split()
    dados = [linha[0],linha[1],linha[2],linha[3]]
    new = list(map(float, dados))
    flor = []
    flor.append(new)
    result = svc.predict(flor)

    if result == 0:
        print('setosa', linha[4])
    elif result == 1:
        print('versicolor', linha[4])
    elif result == 2:
        print('virginica', linha[4] )
