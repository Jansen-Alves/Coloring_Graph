import os
import random
import statistics
import networkx as nx
from baseGrafo import heuristicaGulosa
from utils import ler_grafo_dimacs, ler_solucoes_otimas, salvar_resultados
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------------------------- Parâmetros do AG --------------------------- #
TAMANHO_POPULACAO = 30     
MAX_GERACOES = 60           
TAXA_MUTACAO = 0.15         
TAXA_CROSSOVER = 0.85
ELITISMO = 2                
PORCENTAGEM_HEURISTICA = 0.2  

# --------------------------- Funções do AG --------------------------- #

def inicializar_populacao(G):
    populacao = []
    n_heuristica = int(PORCENTAGEM_HEURISTICA * TAMANHO_POPULACAO)
    for _ in range(n_heuristica):
        cor_heuristica, _ = heuristicaGulosa(G, True)
        populacao.append(dict(cor_heuristica))

    for _ in range(TAMANHO_POPULACAO - n_heuristica):
        individuo = {}
        for node in G.nodes():
            individuo[node] = random.randint(0, len(G))
        populacao.append(individuo)
    return populacao

def avaliar(G, individuo):
    conflitos = sum(1 for u, v in G.edges() if individuo[u] == individuo[v])
    num_cores = len(set(individuo.values()))
    return conflitos * 10 + num_cores

def fitness_valor(G, individuo):
    return avaliar(G, individuo)

def selecao_roleta(G, populacao):
    avaliacoes = [fitness_valor(G, ind) for ind in populacao]
    max_fit = max(avaliacoes)
    fitness_invertido = [max_fit - f + 1 for f in avaliacoes]
    total = sum(fitness_invertido)
    pick = random.uniform(0, total)
    current = 0
    for individuo, fit in zip(populacao, fitness_invertido):
        current += fit
        if current > pick:
            return individuo
    return random.choice(populacao)

def crossover(G, pai1, pai2):
    filho = {}
    for node in G.nodes():
        filho[node] = pai1[node] if random.random() < 0.5 else pai2[node]
    return filho

def mutacao(G, individuo):
    for u, v in G.edges():
        if individuo[u] == individuo[v]:
            no_mutar = random.choice([u, v])
            if random.random() < TAXA_MUTACAO:
                vizinhos = list(G.neighbors(no_mutar))
                cores_vizinhos = set(individuo[vz] for vz in vizinhos)
                nova_cor = next((c for c in range(len(G)) if c not in cores_vizinhos), len(G))
                individuo[no_mutar] = nova_cor
    return individuo

def algoritmo_genetico(G, otima=None, mostrar_grafico=False, max_sem_melhora=25):
    populacao = inicializar_populacao(G)
    melhor_solucao = None
    melhor_fitness = float('inf')
    fitness_geracoes = []
    sem_melhora = 0

    for geracao in range(MAX_GERACOES):
        populacao.sort(key=lambda ind: fitness_valor(G, ind))
        nova_populacao = populacao[:ELITISMO]

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1 = selecao_roleta(G, populacao)
            pai2 = selecao_roleta(G, populacao)
            if random.random() < TAXA_CROSSOVER:
                filho = crossover(G, pai1, pai2)
            else:
                filho = dict(pai1)
            nova_populacao.append(filho)

        for i in range(ELITISMO, len(nova_populacao)):
            mutacao(G, nova_populacao[i])

        populacao = nova_populacao
        populacao.sort(key=lambda ind: fitness_valor(G, ind))

        atual_fitness = fitness_valor(G, populacao[0])
        fitness_geracoes.append(atual_fitness)

        conflitos = sum(1 for u, v in G.edges() if populacao[0][u] == populacao[0][v])
        num_cores = len(set(populacao[0].values()))

        # Print de progresso (opcional)
        print(f"Geração {geracao+1}: Fitness={atual_fitness}, Conflitos={conflitos}, Cores={num_cores}")

        # Critério de parada por melhora
        if atual_fitness < melhor_fitness:
            melhor_solucao = populacao[0]
            melhor_fitness = atual_fitness
            sem_melhora = 0
        else:
            sem_melhora += 1

        # Critério de parada por solução ótima
        if otima is not None and conflitos == 0 and num_cores == otima:
            print(f"Solução ótima ({otima}) encontrada na geração {geracao+1}.")
            break

        # Critério de parada por estagnação
        if sem_melhora >= max_sem_melhora:
            print(f"Parando por estagnação após {max_sem_melhora} gerações sem melhora.")
            break

    conflitos_finais = sum(1 for u, v in G.edges() if melhor_solucao[u] == melhor_solucao[v])
    num_cores = len(set(melhor_solucao.values()))

    if mostrar_grafico:
        plt.figure(figsize=(8, 4))
        plt.plot(fitness_geracoes, marker='o')
        plt.title('Progresso do Fitness ao longo das gerações')
        plt.xlabel('Geração')
        plt.ylabel('Fitness (conflitos*10 + cores)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Visualização do grafo colorido
        cores = [melhor_solucao[n] for n in G.nodes()]
        plt.figure(figsize=(8, 8))
        nx.draw_spring(G, node_color=cores, with_labels=True, cmap=plt.cm.tab20)
        plt.title(f'Coloração final do grafo ({num_cores} cores)')
        plt.show()

    return melhor_solucao, (conflitos_finais, num_cores), fitness_geracoes

def executar_algoritmo(nome_grafo, arquivo, solucoes_otimas, resultados_dir, mostrar_grafico=False):
    G = ler_grafo_dimacs(arquivo)
    avaliacoes = []
    tempos = []
    fitness_hist = []
    otima = solucoes_otimas.get(nome_grafo, None)
    for i in range(3):
        print(f"\n===== {nome_grafo} | Execução {i+1} =====")
        start = time.time()
        _, (conflitos, cores), fitness_geracoes = algoritmo_genetico(G, mostrar_grafico=(mostrar_grafico and i==0))
        end = time.time()
        tempo = end - start
        print(f"Conflitos: {conflitos} | Cores: {cores} | Tempo: {tempo:.2f}s")
        avaliacoes.append((conflitos, cores))
        tempos.append(tempo)
        fitness_hist.append(fitness_geracoes)
        # Critério de parada: encontrou solução ótima conhecida
        if otima is not None and conflitos == 0 and cores == otima:
            print(f"Solução ótima conhecida ({otima}) encontrada na execução {i+1}. Encerrando execuções.")
            break
    conflitos_m = [c for c, _ in avaliacoes]
    cores_m = [k for _, k in avaliacoes]
    media_conflitos = sum(conflitos_m) / len(conflitos_m)
    media_cores = sum(cores_m) / len(cores_m)
    dp_cores = statistics.stdev(cores_m) if len(cores_m) > 1 else 0
    melhor_encontrado = min(cores_m)
    salvar_resultados(
        nome_grafo,
        avaliacoes,
        media_conflitos,
        media_cores,
        dp_cores,
        melhor_encontrado,
        otima,
        resultados_dir
    )
    print(f"[✓] {nome_grafo}: Melhor={melhor_encontrado}, Ótima={otima}, DP={dp_cores:.2f}, Tempo médio={statistics.mean(tempos):.2f}s")
    # Visualização do progresso médio das execuções
    if mostrar_grafico:
        plt.figure(figsize=(8, 4))
        for i, hist in enumerate(fitness_hist):
            plt.plot(hist, label=f'Execução {i+1}')
        plt.title(f'Fitness ao longo das gerações - {nome_grafo}')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ------------------------ Execução principal ------------------------ #
def main():
    pasta_grafos = "grafos"
    resultados_dir = "resultados"
    os.makedirs(resultados_dir, exist_ok=True)
    solucoes_otimas = ler_solucoes_otimas("solucoes_otimas.csv")

    # Nome do grafo sem extensão (.col)
    grafo = "le450_25c"

    arquivo = os.path.join(pasta_grafos, grafo + ".col")
    if not os.path.exists(arquivo):
        print(f"Grafo {grafo}.col não encontrado na pasta {pasta_grafos}.")
        return

    executar_algoritmo(
        grafo,
        arquivo,
        solucoes_otimas,
        resultados_dir,
        mostrar_grafico=False  # Coloque True para ver os gráficos
    )

if __name__ == "__main__":
    main()