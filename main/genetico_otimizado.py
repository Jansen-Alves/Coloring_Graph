
import os
import random
import statistics
import time
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from baseGrafo import heuristicaGulosa
from utils import ler_grafo_dimacs, ler_solucoes_otimas, salvar_resultados

TAMANHO_POPULACAO = 200
MAX_GERACOES = 300
TAXA_MUTACAO_BASE = 0.3
TAXA_CROSSOVER = 0.85
ELITISMO = 2
PORCENTAGEM_HEURISTICA = 0.5
MAX_SEM_MELHORA = 200
PESO_CONFLITO = 10000

def inicializar_populacao(G, max_cores):
    populacao = []
    n_heuristica = int(PORCENTAGEM_HEURISTICA * TAMANHO_POPULACAO)
    for _ in range(n_heuristica):
        coloracao, _ = heuristicaGulosa(G, aleatorio=True)
        vetor = np.array([coloracao[v] % max_cores for v in sorted(G.nodes())])
        populacao.append(vetor)
    for _ in range(TAMANHO_POPULACAO - n_heuristica):
        individuo = np.random.randint(0, max_cores, size=len(G.nodes()))
        populacao.append(individuo)
    return populacao

def avaliar_np(G_edges, individuo, peso_conflito=PESO_CONFLITO):
    conflitos = sum(1 for u, v in G_edges if individuo[u] == individuo[v])
    num_cores = len(set(individuo))
    return conflitos * peso_conflito + num_cores

def avaliar_populacao_paralela(G_edges, populacao):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda ind: avaliar_np(G_edges, ind), populacao))

def crossover(G, pai1, pai2, max_cores):
    filho = np.zeros(len(pai1), dtype=int)
    for v in range(len(pai1)):
        cor1 = pai1[v]
        cor2 = pai2[v]
        vizinhos = list(G.neighbors(v))
        em_conflito1 = any(pai1[v] == pai1[u] for u in vizinhos)
        em_conflito2 = any(pai2[v] == pai2[u] for u in vizinhos)

        if not em_conflito1 and em_conflito2:
            filho[v] = cor1
        elif em_conflito1 and not em_conflito2:
            filho[v] = cor2
        else:
            filho[v] = random.choice([cor1, cor2])
        filho[v] %= max_cores
    return filho

def mutacao(G, individuo, max_cores):
    novo = individuo.copy()
    conflitos = [v for v in G.nodes() if any(novo[v] == novo[u] for u in G.neighbors(v))]
    v = random.choice(conflitos if conflitos else list(G.nodes()))
    vizinhos = set(novo[u] for u in G.neighbors(v))
    cores_disponiveis = [c for c in range(max_cores) if c not in vizinhos]
    novo[v] = random.choice(cores_disponiveis) if cores_disponiveis else random.randint(0, max_cores - 1)
    return novo

def busca_local_gulosa(G, individuo, max_cores):
    novo = individuo.copy()
    for v in G.nodes():
        vizinhos = set(novo[u] for u in G.neighbors(v))
        for cor in range(max_cores):
            if cor not in vizinhos:
                novo[v] = cor
                break
    return novo

def pos_processamento_reducao_cores(individuo):
    cores = list(set(individuo))
    mapeamento = {c: i for i, c in enumerate(cores)}
    return np.array([mapeamento[c] for c in individuo])

def algoritmo_genetico(G, k_max, otima=None):
    G_edges = list(G.edges())
    populacao = inicializar_populacao(G, k_max)
    melhor_solucao = None
    melhor_fitness = float('inf')
    sem_melhora = 0
    taxa_mutacao = TAXA_MUTACAO_BASE

    for geracao in range(MAX_GERACOES):
        avaliacoes = avaliar_populacao_paralela(G_edges, populacao)
        populacao = [x for _, x in sorted(zip(avaliacoes, populacao), key=lambda x: x[0])]
        nova_pop = populacao[:ELITISMO]

        while len(nova_pop) < TAMANHO_POPULACAO:
            if random.random() < TAXA_CROSSOVER:
                p1 = random.choice(populacao[:20])
                p2 = random.choice(populacao[:20])
                filho = crossover(G, p1, p2, k_max)
            else:
                filho = populacao[random.randint(0, TAMANHO_POPULACAO - 1)].copy()
            nova_pop.append(filho)

        for i in range(ELITISMO, TAMANHO_POPULACAO):
            if random.random() < taxa_mutacao:
                nova_pop[i] = mutacao(G, nova_pop[i], k_max)

        if geracao % 10 == 0 and geracao > 0:
            for i in range(TAMANHO_POPULACAO - 20, TAMANHO_POPULACAO):
                nova_pop[i] = np.random.randint(0, k_max, size=len(G.nodes()))

        for i in range(ELITISMO):
            nova_pop[i] = busca_local_gulosa(G, nova_pop[i], k_max)

        populacao = nova_pop
        melhor = populacao[0]
        fitness = avaliar_np(G_edges, melhor)
        conflitos = sum(1 for u, v in G_edges if melhor[u] == melhor[v])
        num_cores = len(set(melhor))

        print(f"Geração {geracao+1}: Fitness={fitness}, Conflitos={conflitos}, Cores={num_cores}")

        if conflitos == 0:
            melhor_solucao = pos_processamento_reducao_cores(melhor)
            melhor_fitness = num_cores
            break

        sem_melhora += 1
        if sem_melhora % 20 == 0:
            taxa_mutacao = min(1.0, taxa_mutacao + 0.1)
            print(f"Aumentando taxa de mutação para {taxa_mutacao:.2f}")

        if sem_melhora >= MAX_SEM_MELHORA:
            print("Parando por estagnação.")
            break

    if melhor_solucao is None:
        melhor_solucao = pos_processamento_reducao_cores(populacao[0])

    conflitos_finais = sum(1 for u, v in G_edges if melhor_solucao[u] == melhor_solucao[v])
    num_cores = len(set(melhor_solucao))
    return melhor_solucao, (conflitos_finais, num_cores), []

def main():
    pasta_grafos = "grafos"
    resultados_dir = "resultados"
    os.makedirs(resultados_dir, exist_ok=True)
    solucoes_otimas = ler_solucoes_otimas("solucoes_otimas.csv")

    grafo_nome = "r250.5"
    arquivo = os.path.join(pasta_grafos, grafo_nome + ".col")
    otima = solucoes_otimas.get(grafo_nome, None)
    G = ler_grafo_dimacs(arquivo)

    coloracao_gulosa, cor_gulosa = heuristicaGulosa(G, aleatorio=True)
    k_atual = max(coloracao_gulosa.values()) + 1
    melhor_k = k_atual

    avaliacoes = []
    tempos = []

    for execucao in range(3):
        while k_atual >= 1:
            print(f"\n=== Tentando com {k_atual} cores ===")
            start = time.time()
            _, (conflitos, num_cores), _ = algoritmo_genetico(G, k_max=k_atual, otima=otima)
            end = time.time()
            tempo = end - start

            avaliacoes.append((conflitos, num_cores))
            tempos.append(tempo)

            print(f"Conflitos: {conflitos} | Cores: {num_cores} | Tempo: {tempo:.2f}s")

            if conflitos == 0:
                melhor_k = num_cores
                if otima is not None and num_cores == otima:
                    salvar_resultados(grafo_nome, avaliacoes, [k for c, k in avaliacoes if c == 0],
                                      melhor_k, melhor_k, 0.0, otima, resultados_dir)
                    print(f"[✓] {grafo_nome}: Melhor={melhor_k}, Ótima={otima}, Tempo médio={statistics.mean(tempos):.2f}s")
                    return
                k_atual = melhor_k - 1
            else:
                print(f"Falhou com {k_atual} cores. Melhor anterior: {melhor_k} cores sem conflitos.")
                break

        cores_m = [k for c, k in avaliacoes if c == 0]
        melhor = min(cores_m) if cores_m else melhor_k
        media = statistics.mean(cores_m) if cores_m else melhor_k
        desvio = statistics.stdev(cores_m) if len(cores_m) > 1 else 0.0

        salvar_resultados(grafo_nome, avaliacoes, cores_m, melhor, media, desvio, otima, resultados_dir)

    print(f"[✓] {grafo_nome}: Melhor={melhor}, Ótima={otima}, DP={desvio:.2f}, Tempo médio={statistics.mean(tempos):.2f}s")

if __name__ == "__main__":
    main()
