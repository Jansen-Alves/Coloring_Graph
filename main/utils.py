# utils.py

import networkx as nx
import csv
import os
import statistics

def ler_grafo_dimacs(caminho_arquivo):
    """
    Lê um arquivo .col no formato DIMACS e retorna um grafo networkx.
    """
    G = nx.Graph()
    with open(caminho_arquivo, 'r') as arquivo:
        for linha in arquivo:
            if linha.startswith('e'):
                _, u, v = linha.strip().split()
                G.add_edge(int(u), int(v))
    return G


def ler_solucoes_otimas(caminho_csv):
    """
    Lê o CSV com soluções ótimas e retorna um dicionário: {nome_grafo: valor_ótimo}.
    """
    solucoes = {}
    with open(caminho_csv, mode='r', encoding='utf-8') as arquivo_csv:
        leitor = csv.reader(arquivo_csv)
        next(leitor)  # pula cabeçalho
        for linha in leitor:
            nome_grafo, solucao = linha
            solucoes[nome_grafo] = int(solucao)
    return solucoes


def salvar_resultados(nome_grafo, execucoes, cores_usadas, melhor, media, desvio, otima = None, pasta='resultados'):
    """
    Salva os resultados de execuções em um CSV.
    
    Parâmetros:
    - nome_grafo: nome do grafo (sem extensão)
    - execucoes: lista de tuplas (conflitos, cores_usadas)
    - pasta: pasta onde salvar os arquivos
    - otima: solução ótima conhecida (opcional)
    """
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    caminho = os.path.join(pasta, f'{nome_grafo}.csv')
    with open(caminho, mode='w', newline='', encoding='utf-8') as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(['execucao', 'conflitos', 'cores_usadas'])

        cores = []
        for i, (conflitos, num_cores) in enumerate(execucoes, start=1):
            escritor.writerow([i, conflitos, num_cores])
            cores.append(num_cores)

        melhor = min(cores)
        media = statistics.mean(cores)
        desvio = statistics.stdev(cores) if len(cores) > 1 else 0.0

        escritor.writerow([])
        escritor.writerow(['melhor_solucao', melhor])
        escritor.writerow(['media_cores', f"{media:.2f}"])
        escritor.writerow(['desvio_padrao', f"{desvio:.2f}"])
        if otima is not None:
            escritor.writerow(['solucao_otima', otima])