import os
import random
import statistics
import networkx as nx
from baseGrafo import heuristicaGulosa
from utils import ler_grafo_dimacs, ler_solucoes_otimas, salvar_resultados

# --------------------------- Parâmetros do AG --------------------------- #
TAMANHO_POPULACAO = 50
MAX_GERACOES = 200
TAXA_MUTACAO = 0.1
TAXA_CROSSOVER = 0.8
ELITISMO = 5  # número de melhores soluções mantidas
PORCENTAGEM_HEURISTICA = 0.1  # % da população inicial vindo da heurística gulosa

# --------------------------- Funções do AG --------------------------- #

def inicializar_populacao(G):
    populacao = []
    n_heuristica = int(PORCENTAGEM_HEURISTICA * TAMANHO_POPULACAO)
    cor_heuristica, _ = heuristicaGulosa(G)

    for _ in range(n_heuristica):
        populacao.append(dict(cor_heuristica))

    for _ in range(TAMANHO_POPULACAO - n_heuristica):
        individuo = {}
        for node in G.nodes():
            individuo[node] = random.randint(0, len(G))
        populacao.append(individuo)

    print("\n[População Inicial] Amostra de 3 indivíduos (10 genes):")
    for ind in populacao[:3]:
        print(dict(list(ind.items())[:10]))

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

def algoritmo_genetico(G):
    populacao = inicializar_populacao(G)
    melhor_solucao = None
    melhor_fitness = float('inf')

    for geracao in range(MAX_GERACOES):
        populacao.sort(key=lambda ind: fitness_valor(G, ind))
        nova_populacao = populacao[:ELITISMO]

        pais = [selecao_roleta(G, populacao) for _ in range((TAMANHO_POPULACAO - ELITISMO) * 2)]
        print("\n[Seleção] Fitness dos 3 primeiros pais:", [avaliar(G, p) for p in pais[:3]])

        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1 = selecao_roleta(G, populacao)
            pai2 = selecao_roleta(G, populacao)

            if random.random() < TAXA_CROSSOVER:
                filho = crossover(G, pai1, pai2)
            else:
                filho = dict(pai1)

            nova_populacao.append(filho)

        print("\n[Crossover] Primeiros 3 filhos (10 genes):")
        for filho in nova_populacao[ELITISMO:ELITISMO+3]:
            print(dict(list(filho.items())[:10]))

        print("\n[Mutação] Verificando mudanças nos 3 primeiros filhos após mutação:")
        for i in range(ELITISMO, ELITISMO + 3):
            original = nova_populacao[i].copy()
            mutado = mutacao(G, nova_populacao[i])
            mudou = original != mutado
            print(f"Filho {i-ELITISMO+1}: {'MUDOU' if mudou else 'NÃO MUDOU'}")

        populacao = nova_populacao
        populacao.sort(key=lambda ind: fitness_valor(G, ind))

        if fitness_valor(G, populacao[0]) < melhor_fitness:
            melhor_solucao = populacao[0]
            melhor_fitness = fitness_valor(G, melhor_solucao)

        if avaliar(G, populacao[0]) < 1000:
            break

    conflitos_finais = sum(1 for u, v in G.edges() if melhor_solucao[u] == melhor_solucao[v])
    num_cores = len(set(melhor_solucao.values()))

    print("\n[Execução Final] Cores usadas por 10 indivíduos:", [len(set(ind.values())) for ind in populacao[:10]])

    return melhor_solucao, (conflitos_finais, num_cores)

# ------------------------ Execução principal ------------------------ #
def main():
    pasta_grafos = "grafos"
    resultados_dir = "resultados"
    os.makedirs(resultados_dir, exist_ok=True)

    solucoes_otimas = ler_solucoes_otimas("solucoes_otimas.csv")

    for arquivo in os.listdir(pasta_grafos):
        if arquivo.endswith(".col"):
            nome_grafo = arquivo[:-4]
            G = ler_grafo_dimacs(os.path.join(pasta_grafos, arquivo))

            avaliacoes = []
            for i in range(3):
                print("\n" + "="*40)
                print(f"===== Execução {i+1} =====")
                grau_max = max(dict(G.degree()).values())
                num_cores = random.randint(3, grau_max)
                print(f"Seed randômica: {random.random():.5f}")
                print(f"Número inicial de cores (estimado): {num_cores}")

                _, (conflitos, cores) = algoritmo_genetico(G)
                avaliacoes.append((conflitos, cores))

            conflitos_m = [c for c, _ in avaliacoes]
            cores_m = [k for _, k in avaliacoes]
            media_conflitos = sum(conflitos_m) / len(conflitos_m)
            media_cores = sum(cores_m) / len(cores_m)
            dp_cores = statistics.stdev(cores_m) if len(cores_m) > 1 else 0

            melhor_encontrado = min(cores_m)
            otima = solucoes_otimas.get(nome_grafo, "Desconhecida")

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
            print(f"[✓] {nome_grafo}: Melhor={melhor_encontrado}, Ótima={otima}, DP={dp_cores:.2f}")

if __name__ == "__main__":
    main()
