"""
Implementação do Baseline (100% das características).
"""

import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
import config


def executar_baseline(X_train, y_train, X_test, y_test):
    """
    Executa baseline com todas as características.
    
    Returns:
        dict: Resultados com acurácia, características, tempos
    """
    print("\n=== BASELINE ===")
    
    tempo_inicio = time.time()
    
    # Treinar com todas características
    modelo = DecisionTreeClassifier(random_state=config.SEMENTE)
    modelo.fit(X_train, y_train)
    acuracia_teste = modelo.score(X_test, y_test)
    
    tempo_treino = time.time() - tempo_inicio
    
    print(f"Acurácia teste: {acuracia_teste:.4f}")
    print(f"Características: {X_train.shape[1]}/{X_train.shape[1]} (100.0%)")
    
    return {
        'metodo': 'Baseline',
        'acuracia': acuracia_teste,
        'porcentagem_caracteristicas': 100.0,
        'tempo_treino': tempo_treino,
        'tempo_busca': 0.0
    }
