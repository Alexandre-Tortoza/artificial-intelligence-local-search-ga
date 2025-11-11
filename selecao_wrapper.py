"""
Implementação de Forward Selection (Wrapper).
"""

import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
import config


def forward_selection(X_train, y_train, X_val, y_val):
    """
    Executa Forward Selection.
    
    Returns:
        array: Máscara binária de características selecionadas
    """
    num_caracteristicas = X_train.shape[1]
    selecionadas = set()
    disponiveis = set(range(num_caracteristicas))
    
    melhor_acuracia = 0
    iteracoes_sem_melhoria = 0
    
    for iteracao in range(config.WRAPPER_MAX_ITERACOES):
        if len(disponiveis) == 0:
            break
        
        melhor_caracteristica = None
        melhor_acuracia_iteracao = melhor_acuracia
        
        # Testar cada característica disponível
        for caracteristica in disponiveis:
            temp_selecionadas = selecionadas | {caracteristica}
            mascara = np.zeros(num_caracteristicas, dtype=bool)
            mascara[list(temp_selecionadas)] = True
            
            # Treinar e avaliar
            X_train_sel = X_train[:, mascara]
            X_val_sel = X_val[:, mascara]
            
            modelo = DecisionTreeClassifier(random_state=config.SEMENTE)
            modelo.fit(X_train_sel, y_train)
            acuracia = modelo.score(X_val_sel, y_val)
            
            if acuracia > melhor_acuracia_iteracao:
                melhor_acuracia_iteracao = acuracia
                melhor_caracteristica = caracteristica
        
        # Adicionar melhor característica
        if melhor_caracteristica is not None:
            selecionadas.add(melhor_caracteristica)
            disponiveis.remove(melhor_caracteristica)
            
            if melhor_acuracia_iteracao > melhor_acuracia:
                melhor_acuracia = melhor_acuracia_iteracao
                iteracoes_sem_melhoria = 0
            else:
                iteracoes_sem_melhoria += 1
            
            if iteracao % 10 == 0:
                print(f"Iteração {iteracao}: {len(selecionadas)} características, "
                      f"acurácia = {melhor_acuracia:.4f}")
        else:
            iteracoes_sem_melhoria += 1
        
        # Parada
        if iteracoes_sem_melhoria >= config.WRAPPER_PARADA_SEM_MELHORIA:
            print(f"Parada antecipada na iteração {iteracao}")
            break
    
    # Criar máscara final
    mascara_final = np.zeros(num_caracteristicas, dtype=int)
    mascara_final[list(selecionadas)] = 1
    
    return mascara_final


def executar_wrapper(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Executa Wrapper Selection completo.
    
    Returns:
        dict: Resultados com acurácia, características, tempos
    """
    print("\n=== WRAPPER SELECTION ===")
    tempo_inicio = time.time()
    
    # Forward selection
    mascara = forward_selection(X_train, y_train, X_val, y_val)
    
    tempo_busca = time.time() - tempo_inicio
    
    # Avaliar no teste
    tempo_inicio_treino = time.time()
    
    # Usar TODOS os dados de treino
    X_train_completo = np.vstack([X_train, X_val])
    y_train_completo = np.hstack([y_train, y_val])
    
    mascara_bool = mascara.astype(bool)
    X_train_sel = X_train_completo[:, mascara_bool]
    X_test_sel = X_test[:, mascara_bool]
    
    modelo_final = DecisionTreeClassifier(random_state=config.SEMENTE)
    modelo_final.fit(X_train_sel, y_train_completo)
    acuracia_teste = modelo_final.score(X_test_sel, y_test)
    
    tempo_treino = time.time() - tempo_inicio_treino
    
    num_selecionadas = mascara.sum()
    porcentagem = 100 * num_selecionadas / len(mascara)
    
    print(f"Acurácia teste: {acuracia_teste:.4f}")
    print(f"Características: {num_selecionadas}/{len(mascara)} ({porcentagem:.1f}%)")
    print(f"Tempo busca: {tempo_busca:.1f}s")
    
    return {
        'metodo': 'Wrapper',
        'acuracia': acuracia_teste,
        'porcentagem_caracteristicas': porcentagem,
        'tempo_treino': tempo_treino,
        'tempo_busca': tempo_busca
    }
