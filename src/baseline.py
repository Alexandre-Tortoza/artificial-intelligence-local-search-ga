import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
import config


def executar_baseline(X_train, y_train, X_test, y_test):
    tempo_inicio = time.time()
    
    modelo = DecisionTreeClassifier(random_state=config.SEMENTE)
    modelo.fit(X_train, y_train)
    acuracia_teste = modelo.score(X_test, y_test)
    
    tempo_treino = time.time() - tempo_inicio
    
    return {
        'metodo': 'Baseline',
        'acuracia': acuracia_teste,
        'porcentagem_caracteristicas': 100.0,
        'tempo_treino': tempo_treino,
        'tempo_busca': 0.0
    }
