import networkx as nx
import matplotlib.pyplot as plt
import algorithmx

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
        "grafoBase2.csv",
        delimiter = ",",
        data = False,
        encoding ='utf-8'
    )
def heuristicaGulosa(G):
    vertices = G.number_of_nodes()
    colorizacao = {node: -1 for node in G.nodes()}
    colorizacao[0]=0
    disponiveis = {node: False for node in G.nodes()}
    lista_vizinhos = {n: {viz: -1 for viz in G.neighbors(n)} for n in G.nodes()}

    for node in range(1, vertices):
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
    
    
    return colorizacao, n_cores