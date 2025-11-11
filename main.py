#!/usr/bin/env -S uv run python

import numpy as np
import pandas as pd
import src.config
from src.utils import carregar_dados, criar_validacao
from src.algoritmo_genetico import executar_ag
from src.selecao_wrapper import executar_wrapper
from src.baseline import executar_baseline


def main():
    
    np.random.seed(src.config.SEMENTE)
    
    X_train, y_train, X_test, y_test = carregar_dados()
    
    X_train_sub, X_val, y_train_sub, y_val = criar_validacao(X_train, y_train)
    
    resultado_ag = executar_ag(X_train_sub, y_train_sub, X_val, y_val, X_test, y_test)
    
    resultado_wrapper = executar_wrapper(X_train_sub, y_train_sub, X_val, y_val, X_test, y_test)
    
    resultado_baseline = executar_baseline(X_train, y_train, X_test, y_test)
    
    df = pd.DataFrame([resultado_ag, resultado_wrapper, resultado_baseline])
    df = df[['metodo', 'acuracia', 'porcentagem_caracteristicas', 'tempo_treino', 'tempo_busca']]
    df.columns = ['Método', 'Acurácia', 'Características (%)', 'Tempo Treino (s)', 'Tempo Busca (s)']
    
    
    df.to_csv('resultados_comparacao.csv', index=False)


if __name__ == '__main__':
    main()
