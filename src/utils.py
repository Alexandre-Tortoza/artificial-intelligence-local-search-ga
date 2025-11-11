import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import config


def carregar_dados():
    dados_treino = pd.read_csv(config.ARQUIVO_TREINO)
    y_train = dados_treino.iloc[:, 0].values
    X_train = dados_treino.iloc[:, 1:].values
    
    dados_teste = pd.read_csv(config.ARQUIVO_TESTE)
    y_test = dados_teste.iloc[:, 0].values
    X_test = dados_teste.iloc[:, 1:].values
    
    return X_train, y_train, X_test, y_test


def criar_validacao(X_train, y_train):
    return train_test_split(
        X_train, y_train,
        test_size=config.PROPORCAO_VALIDACAO,
        random_state=config.SEMENTE,
        stratify=y_train
    )
