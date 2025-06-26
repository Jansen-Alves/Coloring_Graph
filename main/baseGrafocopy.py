import os
import random
from collections import defaultdict

def ler_grafo_dimacs(caminho_arquivo):
    with open(caminho_arquivo, 'r') as f:
        linhas = f.readlines()

    n_vertices = 0
    arestas = []

    for linha in linhas:
        if linha.startswith('p'):
            _, _, n_vertices, _ = linha.strip().split()
            n_vertices = int(n_vertices)
        elif linha.startswith('e'):
            _, u, v = linha.strip().split()
            arestas.append((int(u) - 1, int(v) - 1))  # zero-based index

    grafo = defaultdict(list)
    for u, v in arestas:
        grafo[u].append(v)
        grafo[v].append(u)

    return grafo, n_vertices

def heuristica_construcao_gulosa(grafo, n_vertices, aleatorio=True, vertice_inicial=None):
    cores = [-1] * n_vertices
    vertices = list(range(n_vertices))

    if aleatorio:
        if vertice_inicial is None:
            vertice_inicial = random.choice(vertices)
        if vertice_inicial in vertices:
            vertices.remove(vertice_inicial)
            vertices.insert(0, vertice_inicial)
    else:
        if vertice_inicial is not None and vertice_inicial in vertices:
            vertices.remove(vertice_inicial)
            vertices.insert(0, vertice_inicial)
        # caso contrário, segue a ordem natural do 0 ao n-1

    for u in vertices:
        cores_vizinhos = {cores[v] for v in grafo[u] if cores[v] != -1}
        cor = 0
        while cor in cores_vizinhos:
            cor += 1
        cores[u] = cor

    return cores

def gerar_solucoes_para_genetico(grafo, n_vertices, quantidade=10):
    solucoes = []
    vertices_disponiveis = list(range(n_vertices))
    random.shuffle(vertices_disponiveis)

    for i in range(min(quantidade, n_vertices)):
        vertice_inicial = vertices_disponiveis[i]
        solucao = heuristica_construcao_gulosa(
            grafo, n_vertices, aleatorio=True, vertice_inicial=vertice_inicial
        )
        solucoes.append(solucao)

    return solucoes

def main():
    nome_arquivo = 'le450_25c.col'  # Altere para o nome do grafo na pasta /grafos
    caminho = os.path.join('grafos', nome_arquivo)

    grafo, n_vertices = ler_grafo_dimacs(caminho)

    print("🧊 Simulated Annealing / Busca Tabu (modo fixo):")
    solucao_fixa = heuristica_construcao_gulosa(grafo, n_vertices, aleatorio=False)
    print("Coloração:", solucao_fixa)

    print("\n🎲 Simulated Annealing / Busca Tabu (modo aleatório):")
    solucao_aleatoria = heuristica_construcao_gulosa(grafo, n_vertices, aleatorio=True)
    print("Coloração:", solucao_aleatoria)

    print("\n🧬 Algoritmo Genético (várias soluções):")
    solucoes_genetico = gerar_solucoes_para_genetico(grafo, n_vertices, quantidade=5)
    for i, sol in enumerate(solucoes_genetico):
        print(f"Solução {i+1}: {sol}")

if __name__ == '__main__':
    main()
