import os
import random
import statistics
import time
from baseGrafo import heuristicaGulosa
from utils import ler_grafo_dimacs, ler_solucoes_otimas, salvar_resultados

TAMANHO_POPULACAO = 200
MAX_GERACOES = 600
TAXA_MUTACAO_BASE = 0.4
TAXA_CROSSOVER = 0.85
ELITISMO = 2
PORCENTAGEM_HEURISTICA = 0.5
MAX_SEM_MELHORA = 500
PESO_CONFLITO = 10000  

def inicializar_populacao(G, max_cores):
    populacao = []
    n_heuristica = int(PORCENTAGEM_HEURISTICA * TAMANHO_POPULACAO)
    for _ in range(n_heuristica):
        coloracao, _ = heuristicaGulosa(G, aleatorio=True)
        for v in coloracao:
            coloracao[v] = coloracao[v] % max_cores
        populacao.append(dict(coloracao))
    for _ in range(TAMANHO_POPULACAO - n_heuristica):
        individuo = {node: random.randint(0, max_cores - 1) for node in G.nodes()}
        populacao.append(individuo)
    return populacao

def avaliar(G, individuo):
    conflitos = sum(1 for u, v in G.edges() if individuo[u] == individuo[v])
    num_cores = len(set(individuo.values()))
    return conflitos * PESO_CONFLITO + num_cores

def crossover(G, pai1, pai2, max_cores):
    filho = {}
    for v in G.nodes():
        # Crossover guiado por conflitos: dê preferência ao pai que NÃO está em conflito naquele vértice
        cor1 = pai1[v]
        cor2 = pai2[v]
        em_conflito1 = any(pai1[v] == pai1[u] for u in G.neighbors(v))
        em_conflito2 = any(pai2[v] == pai2[u] for u in G.neighbors(v))

        if not em_conflito1 and em_conflito2:
            filho[v] = cor1
        elif em_conflito1 and not em_conflito2:
            filho[v] = cor2
        else:
            filho[v] = random.choice([cor1, cor2])

        filho[v] %= max_cores
    return filho

def mutacao(G, individuo, max_cores):
    # Prioriza vértices em conflito
    conflitos = [v for v in G.nodes() if any(individuo[v] == individuo[u] for u in G.neighbors(v))]
    if conflitos:
        v = random.choice(conflitos)
        vizinhos = set(individuo[u] for u in G.neighbors(v))
        cores_disponiveis = [cor for cor in range(max_cores) if cor not in vizinhos]
        if cores_disponiveis:
            individuo[v] = random.choice(cores_disponiveis)
        else:
            individuo[v] = random.randint(0, max_cores - 1)
    else:
        v = random.choice(list(G.nodes()))
        vizinhos = set(individuo[u] for u in G.neighbors(v))
        cores_disponiveis = [cor for cor in range(max_cores) if cor not in vizinhos]
        if cores_disponiveis:
            individuo[v] = random.choice(cores_disponiveis)
        else:
            individuo[v] = random.randint(0, max_cores - 1)
    return individuo

def busca_local_gulosa(G, individuo, max_cores):
    for v in G.nodes():
        vizinhos = set(individuo[u] for u in G.neighbors(v))
        for cor in range(max_cores):
            if cor not in vizinhos:
                individuo[v] = cor
                break
    return individuo

def pos_processamento_reducao_cores(G, individuo):
    cores_usadas = list(set(individuo.values()))
    cor_mapeamento = {cor: idx for idx, cor in enumerate(cores_usadas)}
    novo_individuo = {v: cor_mapeamento[individuo[v]] for v in individuo}
    return novo_individuo

def algoritmo_genetico(G, k_max, otima=None):
    populacao = inicializar_populacao(G, k_max)
    melhor_solucao = None
    melhor_fitness = float('inf')
    sem_melhora = 0
    taxa_mutacao = TAXA_MUTACAO_BASE

    for geracao in range(MAX_GERACOES):
        populacao.sort(key=lambda ind: avaliar(G, ind))
        nova_populacao = populacao[:ELITISMO]

        # Crossover
        while len(nova_populacao) < TAMANHO_POPULACAO:
            if random.random() < TAXA_CROSSOVER:
                pai1 = random.choice(populacao[:20])
                pai2 = random.choice(populacao[:20])
                filho = crossover(G, pai1, pai2, k_max)
            else:
                filho = dict(random.choice(populacao))
            nova_populacao.append(filho)

        # Mutação
        for i in range(ELITISMO, TAMANHO_POPULACAO):
            if random.random() < taxa_mutacao:
                nova_populacao[i] = mutacao(G, dict(nova_populacao[i]), k_max)

        # Reinjeção de diversidade a cada 10 gerações
        if geracao % 10 == 0 and geracao > 0:
            num_diversos = int(0.1 * TAMANHO_POPULACAO)
            for i in range(TAMANHO_POPULACAO - num_diversos, TAMANHO_POPULACAO):
                novo = {v: random.randint(0, k_max - 1) for v in G.nodes()}
                nova_populacao[i] = novo

        # Busca local
        for i in range(ELITISMO):
            nova_populacao[i] = busca_local_gulosa(G, dict(nova_populacao[i]), k_max)

        populacao = nova_populacao
        populacao.sort(key=lambda ind: avaliar(G, ind))

        conflitos = sum(1 for u, v in G.edges() if populacao[0][u] == populacao[0][v])
        num_cores = len(set(populacao[0].values()))
        atual_fitness = avaliar(G, populacao[0])

        print(f"Geração {geracao+1}: Fitness={atual_fitness}, Conflitos={conflitos}, Cores={num_cores}")

        if conflitos == 0:
            melhor_solucao = dict(populacao[0])
            melhor_fitness = num_cores
            melhor_solucao = pos_processamento_reducao_cores(G, melhor_solucao)
            break

        # Aumentar taxa de mutação se ficar estagnado
        sem_melhora += 1
        if sem_melhora % 20 == 0:
            taxa_mutacao = min(1.0, taxa_mutacao + 0.1)
            print(f"Aumentando taxa de mutação para {taxa_mutacao:.2f}")

        if sem_melhora >= MAX_SEM_MELHORA:
            print(f"Parando por estagnação após {MAX_SEM_MELHORA} gerações sem melhora.")
            break

    if melhor_solucao is None:
        melhor_solucao = dict(populacao[0])

    conflitos_finais = sum(1 for u, v in G.edges() if melhor_solucao[u] == melhor_solucao[v])
    num_cores = len(set(melhor_solucao.values()))
    return melhor_solucao, (conflitos_finais, num_cores), []

def main():
    pasta_grafos = "grafos"
    resultados_dir = "resultados"
    os.makedirs(resultados_dir, exist_ok=True)
    solucoes_otimas = ler_solucoes_otimas("solucoes_otimas.csv")

    grafo_nome = "le450_25c"  # Nome do grafo a ser processado
    arquivo = os.path.join(pasta_grafos, grafo_nome + ".col")
    otima = solucoes_otimas.get(grafo_nome, None)
    G = ler_grafo_dimacs(arquivo)

    avaliacoes = []
    tempos = []

    coloracao_gulosa, cor_gulosa = heuristicaGulosa(G, aleatorio=True)
    k_atual = max(coloracao_gulosa.values()) + 1
    melhor_k = k_atual

    for execucao in range(3):  # Número de execuções por grafo
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
                # Se encontrou a solução ótima, encerra tudo!
                if otima is not None and num_cores == otima:
                    print(f"Solução ótima ({otima}) encontrada! Encerrando execuções.")
                    salvar_resultados(
                        grafo_nome,
                        avaliacoes,
                        [k for _, k in avaliacoes if _ == 0],
                        melhor_k,
                        melhor_k,
                        0.0,
                        otima,
                        resultados_dir
                    )
                    print(f"[✓] {grafo_nome}: Melhor={melhor_k}, Ótima={otima}, Tempo médio={statistics.mean(tempos):.2f}s")
                    return  # <-- encerra o main imediatamente
                k_atual = melhor_k - 1
            else:
                print(f"Falhou com {k_atual} cores. Melhor anterior: {melhor_k} cores sem conflitos.")
                break

        cores_m = [k for _, k in avaliacoes if _ == 0]
        melhor = min(cores_m) if cores_m else melhor_k
        media = statistics.mean(cores_m) if cores_m else melhor_k
        desvio = statistics.stdev(cores_m) if len(cores_m) > 1 else 0.0

        salvar_resultados(
            grafo_nome,
            avaliacoes,
            cores_m,
            melhor,
            media,
            desvio,
            otima,
            resultados_dir
        )

    print(f"[✓] {grafo_nome}: Melhor={melhor}, Ótima={otima}, DP={desvio:.2f}, Tempo médio={statistics.mean(tempos):.2f}s")

if __name__ == "__main__":
    main()
