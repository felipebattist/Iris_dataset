from sklearn import datasets		
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('iris.csv')

new = list(map(float,input().split(' ')))
iris = datasets.load_iris()

x = iris.data
y = iris.target
C = 1.0
flor = []
flor.append(new)
svc = svm.SVC(kernel='linear', C=C).fit(x, y)

result = svc.predict(flor)
if result == 0:
    print('setosa')
elif result == 1:
    print('versicolor')
elif result == 2:
    print('virginica')
