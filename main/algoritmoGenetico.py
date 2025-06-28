import os
import random
import statistics
import time
from baseGrafo import heuristicaGulosa
from utils import ler_grafo_dimacs, ler_solucoes_otimas, salvar_resultados

TAMANHO_POPULACAO = 200
MAX_GERACOES = 600
TAXA_MUTACAO_BASE = 0.2
TAXA_CROSSOVER = 0.85
ELITISMO = 10
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

def recolore_guloso_reduzindo(G, individuo, cores_inicial):
    # Tenta recolorir usando heurística gulosa, reduzindo o número de cores até não conseguir mais
    melhor_coloracao = dict(individuo)
    melhor_cores = cores_inicial
    for k in range(cores_inicial - 1, 0, -1):
        coloracao, _ = heuristicaGulosa(G, aleatorio=True)
        # Força a coloração a usar no máximo k cores
        for v in coloracao:
            coloracao[v] = coloracao[v] % k
        conflitos = sum(1 for u, v in G.edges() if coloracao[u] == coloracao[v])
        if conflitos == 0:
            melhor_coloracao = dict(coloracao)
            melhor_cores = k
        else:
            break
    return melhor_coloracao, melhor_cores

def algoritmo_genetico(G, k_max, otima=None):
    populacao = inicializar_populacao(G, k_max)
    melhor_solucao = None
    melhor_fitness = float('inf')
    sem_melhora = 0
    taxa_mutacao = TAXA_MUTACAO_BASE

    menor_conflitos = None  # Novo: para controlar o print

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

        # Reinjeção de diversidade a cada 5 gerações
        if geracao % 20 == 0 and geracao > 0:
            num_diversos = int(0.2 * TAMANHO_POPULACAO)
            for i in range(TAMANHO_POPULACAO - num_diversos, TAMANHO_POPULACAO):
                novo = {v: random.randint(0, k_max - 1) for v in G.nodes()}
                nova_populacao[i] = novo

        # Busca local em x% da população
        for i in range(int(0.1 * TAMANHO_POPULACAO)):
            nova_populacao[i] = busca_local_gulosa(G, dict(nova_populacao[i]), k_max)

        populacao = nova_populacao
        populacao.sort(key=lambda ind: avaliar(G, ind))

        conflitos = sum(1 for u, v in G.edges() if populacao[0][u] == populacao[0][v])
        num_cores = len(set(populacao[0].values()))
        atual_fitness = avaliar(G, populacao[0])

        # Imprime se diminuiu o número de conflitos
        if menor_conflitos is None or conflitos < menor_conflitos:
            print(f"Geração {geracao+1}: Fitness={atual_fitness}, Conflitos={conflitos}, Cores={num_cores}")
            menor_conflitos = conflitos

        if conflitos == 0:
            melhor_solucao = dict(populacao[0])
            melhor_fitness = num_cores
            melhor_solucao = pos_processamento_reducao_cores(G, melhor_solucao)
            melhor_solucao, melhor_cores = recolore_guloso_reduzindo(G, melhor_solucao, num_cores)
            break

        # Aumenta a taxa de mutação se ficar estagnado
        sem_melhora += 1
        if sem_melhora % 20 == 0:
            taxa_anterior = taxa_mutacao
            taxa_mutacao = min(1.0, taxa_mutacao + 0.1)
            if taxa_anterior < 1.0 and taxa_mutacao > taxa_anterior:
                print(f"Aumentando taxa de mutação para {taxa_mutacao:.2f}")

        if sem_melhora >= MAX_SEM_MELHORA:
            print(f"Parando por estagnação após {MAX_SEM_MELHORA} gerações sem melhora.")
            break

    if melhor_solucao is None:
        melhor_solucao = dict(populacao[0])

    conflitos_finais = sum(1 for u, v in G.edges() if melhor_solucao[u] == melhor_solucao[v])
    num_cores = len(set(melhor_solucao.values()))
    return melhor_solucao, (conflitos_finais, num_cores), []

def salvar_resultados(
    grafo_nome,
    execucoes,  
    cores_sem_conflito,
    melhor,
    media,
    desvio,
    otima,
    resultados_dir
):
    arquivo = os.path.join(resultados_dir, f"{grafo_nome}.csv")
    with open(arquivo, "w", encoding="utf-8") as f:
        f.write("execucao,conflitos,cores_usadas\n")
        for execucao, conflitos, num_cores in execucoes:
            f.write(f"{execucao},{conflitos},{num_cores}\n")
        f.write(f"\nmelhor_solucao,{melhor}\n")
        f.write(f"media_cores,{media:.2f}\n")
        f.write(f"desvio_padrao,{desvio:.2f}\n")
        if otima is not None:
            f.write(f"solucao_otima,{otima}\n")

def main():
    pasta_grafos = "grafos"
    resultados_dir = "resultados"
    os.makedirs(resultados_dir, exist_ok=True)
    solucoes_otimas = ler_solucoes_otimas("solucoes_otimas.csv")

    grafo_nome = "queen11_11"  # Nome do grafo a ser processado
    arquivo = os.path.join(pasta_grafos, grafo_nome + ".col")
    otima = solucoes_otimas.get(grafo_nome, None)
    G = ler_grafo_dimacs(arquivo)

    print(f"\n=== Iniciando processamento do grafo: {grafo_nome} ===")

    coloracao_gulosa, cor_gulosa = heuristicaGulosa(G, aleatorio=True)
    k_atual = max(coloracao_gulosa.values()) + 1
    melhor_k = k_atual

    melhores_execucoes = []
    tempos = []

    for execucao in range(3):  # Número de execuções por grafo
        k_atual = max(coloracao_gulosa.values()) + 1
        melhor_k = k_atual
        melhor_sem_conflito = None
        tempo_execucao = 0

        while k_atual >= 1:
            print(f"\n=== Tentando com {k_atual} cores ===")
            start = time.time()
            solucao, (conflitos, num_cores), _ = algoritmo_genetico(G, k_max=k_atual, otima=otima)
            end = time.time()
            tempo = end - start
            tempo_execucao += tempo

            print(f"Conflitos: {conflitos} | Cores: {num_cores} | Tempo: {tempo:.2f}s")

            if conflitos == 0:
                if melhor_sem_conflito is None or num_cores < melhor_sem_conflito[1]:
                    melhor_sem_conflito = (conflitos, num_cores)
                melhor_k = num_cores
                if otima is not None and num_cores == otima:
                    print(f"Solução ótima ({otima}) encontrada nesta execução!")
                    break  
                k_atual = melhor_k - 1
            else:
                print(f"Falhou com {k_atual} cores. Melhor anterior: {melhor_k} cores sem conflitos.")
                break

        melhores_execucoes.append(melhor_sem_conflito)
        tempos.append(tempo_execucao)


    avaliacoes_csv = []
    for i, execucao in enumerate(melhores_execucoes):
        if execucao is not None:
            avaliacoes_csv.append((i+1, execucao[0], execucao[1]))

    if avaliacoes_csv:
        melhor = min(k for _, _, k in avaliacoes_csv)
        media = statistics.mean(k for _, _, k in avaliacoes_csv)
        desvio = statistics.stdev([k for _, _, k in avaliacoes_csv]) if len(avaliacoes_csv) > 1 else 0.0
    else:
        melhor = media = desvio = 0

    salvar_resultados(
        grafo_nome,
        avaliacoes_csv,
        [k for _, _, k in avaliacoes_csv],
        melhor,
        media,
        desvio,
        otima,
        resultados_dir
    )

    print(f"[✓] {grafo_nome}: Melhor={melhor}, Ótima={otima}, DP={desvio:.2f}, Tempo médio={statistics.mean(tempos):.2f}s")

if __name__ == "__main__":
    main()
