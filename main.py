#!/usr/bin/env -S uv run python

"""
Script principal para execução do projeto TDE3.

Este script:
1. Carrega os dados MNIST
2. Executa Algoritmo Genético
3. Executa Wrapper Selection
4. Executa Baseline
5. Compara e salva resultados
"""

import numpy as np
import pandas as pd
import config
from utils import carregar_dados, criar_validacao
from algoritmo_genetico import executar_ag
from selecao_wrapper import executar_wrapper
from baseline import executar_baseline


def main():
    """Função principal."""
    
    np.random.seed(config.SEMENTE)
    
    X_train, y_train, X_test, y_test = carregar_dados()
    
    X_train_sub, X_val, y_train_sub, y_val = criar_validacao(X_train, y_train)
    print(f"Validação: {X_train_sub.shape[0]} treino, {X_val.shape[0]} validação")
    
    resultado_ag = executar_ag(X_train_sub, y_train_sub, X_val, y_val, X_test, y_test)
    
    resultado_wrapper = executar_wrapper(X_train_sub, y_train_sub, X_val, y_val, X_test, y_test)
    
    resultado_baseline = executar_baseline(X_train, y_train, X_test, y_test)
    print("\n")
    print("Resultados")
    print("\n")
    
    df = pd.DataFrame([resultado_ag, resultado_wrapper, resultado_baseline])
    df = df[['metodo', 'acuracia', 'porcentagem_caracteristicas', 'tempo_treino', 'tempo_busca']]
    df.columns = ['Método', 'Acurácia', 'Características (%)', 'Tempo Treino (s)', 'Tempo Busca (s)']
    
    print(df.to_string(index=False))
    
    df.to_csv('resultados_comparacao.csv', index=False)
    print("\nResultados salvos em: resultados_comparacao.csv")
    
    print("\n")
    print("Analise")
    print("\n")
    
    melhor_acuracia = df['Acurácia'].max()
    metodo_melhor = df.loc[df['Acurácia'].idxmax(), 'Método']
    print(f"Melhor acurácia: {metodo_melhor} ({melhor_acuracia:.4f})")
    
    menor_caracteristicas = df['Características (%)'].min()
    metodo_menor = df.loc[df['Características (%)'].idxmin(), 'Método']
    print(f"Menor uso de características: {metodo_menor} ({menor_caracteristicas:.1f}%)")
    
    acuracia_ag = df.loc[df['Método'] == 'AG', 'Acurácia'].values[0]
    acuracia_baseline = df.loc[df['Método'] == 'Baseline', 'Acurácia'].values[0]
    carac_ag = df.loc[df['Método'] == 'AG', 'Características (%)'].values[0]
    
    diferenca = acuracia_ag - acuracia_baseline
    print(f"\nTrade-off AG vs Baseline:")
    print(f"  Diferença de acurácia: {diferenca:+.4f}")
    print(f"  Redução de características: {100-carac_ag:.1f}%")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
