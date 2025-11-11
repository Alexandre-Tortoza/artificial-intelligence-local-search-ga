"""
Implementação do Algoritmo Genético para seleção de características.
"""

import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
import config


def criar_populacao_inicial(tamanho_populacao, numero_caracteristicas):
    """Cria população inicial aleatória."""
    return np.random.randint(0, 2, size=(tamanho_populacao, numero_caracteristicas))


def calcular_aptidao(individuo, X_train, y_train, X_val, y_val):
    """
    Calcula aptidão do indivíduo.
    
    Formula: 0.7 * acuracia + 0.3 * (1 - proporcao_caracteristicas)
    """
    num_selecionadas = individuo.sum()
    
    if num_selecionadas == 0:
        return 0.0
    
    # Selecionar características
    mascara = individuo.astype(bool)
    X_train_sel = X_train[:, mascara]
    X_val_sel = X_val[:, mascara]
    
    # Treinar e avaliar
    modelo = DecisionTreeClassifier(random_state=config.SEMENTE)
    modelo.fit(X_train_sel, y_train)
    acuracia = modelo.score(X_val_sel, y_val)
    
    # Calcular aptidão
    proporcao = num_selecionadas / len(individuo)
    aptidao = config.AG_PESO_ACURACIA * acuracia + config.AG_PESO_CARACTERISTICAS * (1 - proporcao)
    
    return aptidao


def selecao_torneio(populacao, aptidoes):
    """Seleciona um indivíduo por torneio."""
    indices = np.random.choice(len(populacao), size=config.AG_TAMANHO_TORNEIO, replace=False)
    melhor_idx = indices[np.argmax(aptidoes[indices])]
    return populacao[melhor_idx].copy()


def cruzamento_uniforme(pai1, pai2):
    """Realiza cruzamento uniforme."""
    if np.random.random() > config.AG_TAXA_CRUZAMENTO:
        return pai1.copy(), pai2.copy()
    
    mascara = np.random.randint(0, 2, len(pai1)).astype(bool)
    filho1 = np.where(mascara, pai1, pai2)
    filho2 = np.where(mascara, pai2, pai1)
    
    return filho1, filho2


def mutacao(individuo):
    """Aplica mutação bit flip."""
    for i in range(len(individuo)):
        if np.random.random() < config.AG_TAXA_MUTACAO:
            individuo[i] = 1 - individuo[i]
    return individuo


def executar_ag(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Executa Algoritmo Genético completo.
    
    Returns:
        dict: Resultados com acurácia, características, tempos
    """
    print("\n=== ALGORITMO GENÉTICO ===")
    tempo_inicio = time.time()
    
    num_caracteristicas = X_train.shape[1]
    
    # Criar população inicial
    populacao = criar_populacao_inicial(config.AG_POPULACAO, num_caracteristicas)
    
    melhor_individuo = None
    melhor_aptidao = 0
    geracoes_sem_melhoria = 0
    
    # Loop evolutivo
    for geracao in range(config.AG_GERACOES):
        # Avaliar população
        aptidoes = np.array([calcular_aptidao(ind, X_train, y_train, X_val, y_val) 
                            for ind in populacao])
        
        # Melhor da geração
        idx_melhor = np.argmax(aptidoes)
        if aptidoes[idx_melhor] > melhor_aptidao:
            melhor_aptidao = aptidoes[idx_melhor]
            melhor_individuo = populacao[idx_melhor].copy()
            geracoes_sem_melhoria = 0
        else:
            geracoes_sem_melhoria += 1
        
        # Log
        if geracao % 10 == 0:
            print(f"Geração {geracao}: Melhor aptidão = {aptidoes[idx_melhor]:.4f}, "
                  f"Características = {populacao[idx_melhor].sum()}")
        
        # Critério de parada
        if geracoes_sem_melhoria >= config.AG_PARADA_SEM_MELHORIA:
            print(f"Parada antecipada na geração {geracao}")
            break
        
        # Nova geração
        nova_populacao = []
        
        # Elitismo
        num_elite = int(config.AG_TAXA_ELITISMO * config.AG_POPULACAO)
        indices_elite = np.argsort(aptidoes)[-num_elite:]
        for idx in indices_elite:
            nova_populacao.append(populacao[idx])
        
        # Gerar novos indivíduos
        while len(nova_populacao) < config.AG_POPULACAO:
            pai1 = selecao_torneio(populacao, aptidoes)
            pai2 = selecao_torneio(populacao, aptidoes)
            filho1, filho2 = cruzamento_uniforme(pai1, pai2)
            filho1 = mutacao(filho1)
            filho2 = mutacao(filho2)
            nova_populacao.append(filho1)
            if len(nova_populacao) < config.AG_POPULACAO:
                nova_populacao.append(filho2)
        
        populacao = np.array(nova_populacao)
    
    tempo_busca = time.time() - tempo_inicio
    
    # Avaliar melhor solução no teste
    tempo_inicio_treino = time.time()
    
    # Usar TODOS os dados de treino (não só validação)
    mascara = melhor_individuo.astype(bool)
    X_train_completo = np.vstack([X_train, X_val])
    y_train_completo = np.hstack([y_train, y_val])
    X_train_sel = X_train_completo[:, mascara]
    X_test_sel = X_test[:, mascara]
    
    modelo_final = DecisionTreeClassifier(random_state=config.SEMENTE)
    modelo_final.fit(X_train_sel, y_train_completo)
    acuracia_teste = modelo_final.score(X_test_sel, y_test)
    
    tempo_treino = time.time() - tempo_inicio_treino
    
    num_selecionadas = melhor_individuo.sum()
    porcentagem = 100 * num_selecionadas / num_caracteristicas
    
    print(f"Acurácia teste: {acuracia_teste:.4f}")
    print(f"Características: {num_selecionadas}/{num_caracteristicas} ({porcentagem:.1f}%)")
    print(f"Tempo busca: {tempo_busca:.1f}s")
    
    return {
        'metodo': 'AG',
        'acuracia': acuracia_teste,
        'porcentagem_caracteristicas': porcentagem,
        'tempo_treino': tempo_treino,
        'tempo_busca': tempo_busca
    }
