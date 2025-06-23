import networkx as nx
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ===================================================================
# 1. FUNÇÕES DE GERAÇÃO DE GRAFO E HEURÍSTICA INICIAL
# ===================================================================

def grafoImport(n: int) -> nx.Graph:
    """Gera um grafo aleatório e o salva, retornando o objeto do grafo."""
    G = nx.gnp_random_graph(n, 0.3, seed=135) # seed para reprodutibilidade
    n_vertices = G.number_of_nodes()
    n_arestas = G.number_of_edges()

    print(f"Grafo gerado com {n_vertices} vértices e {n_arestas} arestas.")
    
    # Salvar o grafo (opcional, mas mantido da sua função original)
    nx.write_edgelist(G, "grafoBase.csv", delimiter=",", data=False, encoding='utf-8')
    
    # A linha mais importante que estava faltando:
    return G

def heuristicaGulosa(G: nx.Graph) -> Tuple[Dict[int, int], int]:
    """
    Aplica uma coloração gulosa ao grafo.
    Retorna o dicionário da coloração e o número de cores utilizadas.
    """
    vertices = G.number_of_nodes()
    coloracao = {node: -1 for node in G.nodes()}
    
    # Ordena os nós por grau (uma melhoria comum para o guloso)
    nodes_ordenados = sorted(G.nodes(), key=lambda x: G.degree[x], reverse=True)
    
    for node in nodes_ordenados:
        cores_vizinhos = {coloracao[vizinho] for vizinho in G.neighbors(node) if coloracao[vizinho] != -1}
        
        cor_atual = 0
        while True:
            if cor_atual not in cores_vizinhos:
                coloracao[node] = cor_atual
                break
            cor_atual += 1
            
    # CORREÇÃO: O número de cores é o maior índice de cor + 1
    n_cores_usadas = max(coloracao.values()) + 1
    return coloracao, n_cores_usadas

# ===================================================================
# 2. FUNÇÕES DO ALGORITMO DE BUSCA TABU
# ===================================================================

def calcular_conflitos(G: nx.Graph, coloracao: Dict[int, int]) -> int:
    """Calcula o número de arestas com vértices da mesma cor."""
    conflitos = 0
    for u, v in G.edges():
        if coloracao[u] == coloracao[v]:
            conflitos += 1
    return conflitos

def gerar_vizinhos_otimizado(G: nx.Graph, coloracao: Dict[int, int], n_cores: int) -> List[Tuple[Dict[int, int], Tuple[int, int]]]:
    """Gera vizinhança focando apenas em nós com conflitos, tornando o processo mais rápido."""
    vizinhos = []
    nos_conflitantes = {u for u, v in G.edges() if coloracao[u] == coloracao[v]} | \
                       {v for u, v in G.edges() if coloracao[u] == coloracao[v]}

    if not nos_conflitantes:
        return []

    for v in nos_conflitantes:
        cor_atual = coloracao[v]
        for nova_cor in range(n_cores):
            if nova_cor != cor_atual:
                nova_coloracao = coloracao.copy()
                nova_coloracao[v] = nova_cor
                movimento = (v, nova_cor)
                vizinhos.append((nova_coloracao, movimento))
    return vizinhos

def busca_tabu_coloracao(G: nx.Graph, max_iter: int, tabu_tamanho: int, n_cores: int, solucao_inicial: Dict[int, int]):
    """
    Executa a Busca Tabu para encontrar uma coloração com `n_cores` sem conflitos.
    Inicia a partir de uma `solucao_inicial` fornecida.
    """
    atual = solucao_inicial
    custo_atual = calcular_conflitos(G, atual)
    
    # Se a solução inicial já é válida para n_cores, reatribui cores se necessário
    # Isso garante que as cores estejam no intervalo [0, n_cores-1]
    maior_cor = max(atual.values())
    if maior_cor >= n_cores:
        for v, c in atual.items():
            atual[v] = c % n_cores
        custo_atual = calcular_conflitos(G, atual)

    melhor = atual.copy()
    melhor_custo = custo_atual
    lista_tabu = []

    if melhor_custo == 0:
        return melhor, melhor_custo

    for _ in range(max_iter):
        vizinhos = gerar_vizinhos_otimizado(G, atual, n_cores)
        if not vizinhos: # Se não há mais nós conflitantes
             break
        
        melhor_vizinho = None
        menor_custo = float('inf')
        movimento_escolhido = None

        for vizinho, movimento in vizinhos:
            custo = calcular_conflitos(G, vizinho)
            # Critério de aspiração: permite movimento tabu se ele leva a uma solução melhor que a melhor já vista
            if (movimento not in lista_tabu) or (custo < melhor_custo):
                if custo < menor_custo:
                    melhor_vizinho = vizinho
                    menor_custo = custo
                    movimento_escolhido = movimento
        
        if melhor_vizinho is None:
            break # Não encontrou nenhum movimento válido

        atual = melhor_vizinho
        custo_atual = menor_custo
        
        lista_tabu.append(movimento_escolhido)
        if len(lista_tabu) > tabu_tamanho:
            lista_tabu.pop(0)

        if custo_atual < melhor_custo:
            melhor = atual.copy()
            melhor_custo = custo_atual
            if melhor_custo == 0:
                break

    return melhor, melhor_custo

# ===================================================================
# 3. FUNÇÃO ESTRATÉGICA E EXECUÇÃO PRINCIPAL
# ===================================================================

def otimizar_com_busca_tabu(G: nx.Graph, solucao_inicial: Dict[int, int], k_inicial: int, max_iter: int, tabu_tamanho: int):
    """
    Recebe uma solução gulosa e tenta diminuir o número de cores (k)
    usando a Busca Tabu de forma decremental.
    """
    melhor_k_geral = k_inicial
    melhor_coloracao_geral = solucao_inicial.copy()
    print("-" * 30)
    print(f"Iniciando otimização da Busca Tabu com k = {k_inicial - 1}")
    print("-" * 30)
    
    # Loop decremental: começa de k_inicial - 1 e vai até 1
    for k_alvo in range(k_inicial - 1, 0, -1):
        print(f"\n[Tentativa] Buscando solução com {k_alvo} cores...")
        
        coloracao, conflitos = busca_tabu_coloracao(
            G, max_iter, tabu_tamanho, n_cores=k_alvo, solucao_inicial=melhor_coloracao_geral
        )
        
        if conflitos == 0:
            print(f"  -> SUCESSO! Solução válida encontrada com {k_alvo} cores.")
            melhor_k_geral = k_alvo
            melhor_coloracao_geral = coloracao.copy()
        else:
            print(f"  -> FALHA. Não foi possível encontrar solução com {k_alvo} cores.")
            print("      Parando a busca e retornando a melhor solução anterior.")
            break # Se falhou, não adianta tentar com menos cores

    return melhor_k_geral, melhor_coloracao_geral


def executar_completo():
    """Orquestra todo o processo de coloração de grafos."""
    # --- Parâmetros ---
    NUM_VERTICES = 30
    MAX_ITER_TABU = 200
    TAMANHO_LISTA_TABU = 10

    # 1. Gerar o grafo
    G = grafoImport(NUM_VERTICES)

    # 2. Obter solução inicial com heurística gulosa
    solucao_gulosa, k_guloso = heuristicaGulosa(G)
    conflitos_gulosos = calcular_conflitos(G, solucao_gulosa)
    print("\n--- Resultado da Heurística Gulosa ---")
    print(f"Número de cores usado: {k_guloso}")
    print(f"Conflitos iniciais: {conflitos_gulosos}") # Deveria ser 0 se o guloso for bem feito

    # 3. Otimizar a solução com a Busca Tabu (estratégia decremental)
    k_final, coloracao_final = otimizar_com_busca_tabu(
        G, solucao_gulosa, k_guloso, MAX_ITER_TABU, TAMANHO_LISTA_TABU
    )

    # 4. Imprimir resultados finais
    print("\n" + "=" * 30)
    print("--- Resultado Final da Busca Tabu ---")
    print(f"Número mínimo de cores encontrado: {k_final}")
    print("Coloração final dos vértices:")
    print(sorted(coloracao_final.items()))

    # 5. Visualização
    pos = nx.spring_layout(G, seed=42)
    cores_map = [coloracao_final.get(v, 0) for v in G.nodes()]
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=cores_map, cmap=plt.cm.jet, node_size=800, font_color='white')
    plt.title(f'Coloração Final com {k_final} cores (sem conflitos)')
    plt.show()

# Executar tudo
executar_completo()