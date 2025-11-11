"""
Funções auxiliares para carregamento de dados e operações comuns.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import config


def carregar_dados():
    """
    Carrega dados MNIST dos arquivos CSV.
    
    Returns:
        tupla: (X_train, y_train, X_test, y_test)
    """
    # Carregar treino
    dados_treino = pd.read_csv(config.ARQUIVO_TREINO)
    y_train = dados_treino.iloc[:, 0].values
    X_train = dados_treino.iloc[:, 1:].values
    
    # Carregar teste
    dados_teste = pd.read_csv(config.ARQUIVO_TESTE)
    y_test = dados_teste.iloc[:, 0].values
    X_test = dados_teste.iloc[:, 1:].values
    
    print(f"Dados carregados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
    return X_train, y_train, X_test, y_test


def criar_validacao(X_train, y_train):
    """
    Divide dados de treino em treino e validação.
    
    Returns:
        tupla: (X_train_novo, X_val, y_train_novo, y_val)
    """
    return train_test_split(
        X_train, y_train,
        test_size=config.PROPORCAO_VALIDACAO,
        random_state=config.SEMENTE,
        stratify=y_train
    )


def treinar_e_avaliar(X_train, y_train, X_test, y_test, mascara_caracteristicas):
    """
    Treina modelo com características selecionadas e avalia.
    
    Args:
        X_train: Dados de treino completos
        y_train: Rótulos de treino
        X_test: Dados de teste completos
        y_test: Rótulos de teste
        mascara_caracteristicas: Array binário de características selecionadas
    
    Returns:
        float: Acurácia no teste
    """
    # Selecionar características
    mascara = mascara_caracteristicas.astype(bool)
    X_train_sel = X_train[:, mascara]
    X_test_sel = X_test[:, mascara]
    
    # Treinar modelo
    modelo = DecisionTreeClassifier(random_state=config.SEMENTE)
    modelo.fit(X_train_sel, y_train)
    
    # Avaliar
    return modelo.score(X_test_sel, y_test)
