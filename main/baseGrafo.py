import networkx as nx
import matplotlib.pyplot as plt
import algorithmx
import random

def grafoImport(n):
    G = nx.gnp_random_graph(n, 0.3, 135)
    # total de vertices e arestas
    n_vertices = G.number_of_nodes()
    n_arestas = G.number_of_edges()

    print('vertices: ', n_vertices, '\narestas: ', n_arestas)
    pares = G.edges()
    no_1 = []
    no_2 = []
    for u, v in pares:
        no_1.append(u)
        no_2.append(v)

    #for i in range(len(no_1)):
    #    print('par', i,':(',no_1[i],',',no_2[i],')')

    nx.draw_circular(G, with_labels=True)
    nx.write_edgelist(
        G,
        "grafoBase.csv",
        delimiter = ",",
        data = False,
        encoding ='utf-8'
    )

def heuristicaGulosa(G, aleatorio=False):
    vertices = G.number_of_nodes()
    nodes = list(G.nodes())
    if aleatorio:
        random.shuffle(nodes)
    else:
        # Garante que o primeiro vértice seja o mesmo de sempre (0)
        nodes.sort()
    colorizacao = {node: -1 for node in nodes}
    colorizacao[nodes[0]] = 0
    disponiveis = [False] * vertices
    lista_vizinhos = {n: {viz: -1 for viz in G.neighbors(n)} for n in nodes}
    #print(vertices)
    for idx in range(1, vertices):
        node = nodes[idx]
        for viz in lista_vizinhos[node]:
            if colorizacao[viz] != -1:
                disponiveis[colorizacao[viz]] = True

        for color in range(vertices):
            if not disponiveis[color]:
                break

        colorizacao[node] = color
        for viz in lista_vizinhos[node]:
            if colorizacao[viz] != -1:
                disponiveis[colorizacao[viz]] = False

    n_cores = max(colorizacao.values())
    return colorizacao, n_cores