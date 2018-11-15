import warnings
warnings.filterwarnings("ignore")
import networkx as nx 
import json
from pprint import pprint
import time
import io
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import random



def get_object_index(object_index_dict, object_id):
    if object_id not in object_index_dict:
        object_index_dict[object_id] = len(object_index_dict)
    return object_index_dict[object_id]


def preprocess_dlbp_data(in_files):

    citations = open("data/dblp/parsed/paper_reference.tsv", "w+")
    paper_venue = open("data/dblp/parsed/paper_venue.tsv", "w+")
    author_paper = open("data/dblp/parsed/author_paper.tsv", "w+")

    paper_index_dict = {}
    author_index_dict = {}
    venue_index_dict = {}
    paper_title_dict = {}

    i = 0

    for f_name in in_files:
        with open(f_name) as f:
            
            times_start = time.time()

            for line in f:
                data = json.loads(line) # json corresponding to one paper
                # print(data)
                paper_id = data["id"]
                paper_index = get_object_index(paper_index_dict, paper_id)

                if "title" in data:
                    paper_title_dict[paper_id] = data["title"]

                if "references" in data:
                    references = data["references"]
                    for reference_id in references:
                        reference_id = reference_id
                        reference_index = get_object_index(paper_index_dict, reference_id)
                        citations.write(str(paper_index) + "\t" + str(reference_index) + "\n")
                
                if "authors" in data:
                    authors = data["authors"]
                    for author in authors:
                        author = author
                        author_index = get_object_index(author_index_dict, author)
                        author_paper.write(str(author_index) + "\t" + str(paper_index) + "\n")

                if "venue" in data:
                    venue = data["venue"]
                    venue_index = get_object_index(venue_index_dict, venue)
                    paper_venue.write(str(paper_index) + "\t" + str(venue_index) + "\n")
                
                i += 1
                if i % 100000 == 0:
                    print(i, time.time() - times_start)
                    times_start = time.time()
                    # break

    with io.open("author_index.tsv", "w+", encoding="utf-8") as f:
        for author_id in author_index_dict:
            f.write(author_id + "\t" + str(author_index_dict[author_id]) + "\n")

    with io.open("paper_index.tsv", "w+") as f:
        for paper_id in paper_index_dict:
            f.write(paper_id + "\t" + str(paper_index_dict[paper_id]) + "\n")

    with io.open("venue_index.tsv", "w+", encoding="utf-8") as f:
        for venue_id in venue_index_dict:
            f.write(venue_id + "\t" + str(venue_index_dict[venue_id]) + "\n")

    with io.open("paper_title.tsv", "w+", encoding="utf-8") as f:
        for paper_id in paper_title_dict:
            # print(paper_title_dict[paper_id].decode('ascii', 'ignore'), type(paper_title_dict[paper_id].decode('ascii', 'ignore')) )
            # print(paper_title_dict[paper_id], type(paper_title_dict[paper_id]))
            # print(paper_title_dict[paper_id].encode("utf-8"), type(paper_title_dict[paper_id].encode("utf-8")))
            # print(paper_title_dict[paper_id].encode("utf-8"), type(paper_title_dict[paper_id].encode("utf-8")))
            # print(paper_title_dict[paper_id].encode("utf-8")[83])
            f.write(paper_id + "\t" + paper_title_dict[paper_id] + "\n")

def visualize(in_embeddings, out_file):
    f = open(in_embeddings)
    f.readline()
    f.readline()
    elements = []
    X = []
    for line in f:
        toks = line.strip().split(" ")
        elements.append(toks[0][0])
        X.append([float(n) for n in toks[1:]])
    X = np.array(X)
    X_embedded = X
    # X_embedded = TSNE(n_components=2).fit_transform(X)
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=elements)
    plt.tight_layout()
    plt.savefig(out_file)


def load_graph_data(file_name):
    f = open(file_name)
    G = nx.Graph()

    for line in f:
        a, b = line.strip().split("\t")
        G.add_edge(a, b)

    G.remove_edges_from(G.selfloop_edges())
    GC = max(nx.connected_component_subgraphs(G), key=len) # take greatest connected component
    A = nx.adjacency_matrix(GC)
    return GC, A


def split_test_train(G, edge_type, seed, val_size=0.1, test_size=0.1):
    np.random.seed(seed)

    t = time.time()

    # select nodes of required types
    nodes_0 = [n for n in G.nodes if n.startswith(edge_type[0])]
    nodes_1 = [n for n in G.nodes if n.startswith(edge_type[1])]

    print(len(nodes_0), len(nodes_1))

    selected_edges = [e for e in G.edges if (e[0].startswith(edge_type[0]) and e[1].startswith(edge_type[1]))
                      or (e[0].startswith(edge_type[1]) and e[1].startswith(edge_type[0]))]
    selected_edges = [(e[1], e[0]) if e[0] > e[1] else e for e in selected_edges]

    G_edge_type = nx.Graph()
    G_edge_type.add_nodes_from(nodes_0 + nodes_1)
    G_edge_type.add_edges_from(selected_edges)

    G_train = G.copy()
    print ("Is connected",  nx.is_connected(G_train))

    n_edges = G_edge_type.number_of_edges()
    m_test = int(test_size * n_edges)
    m_val = int(val_size * n_edges)


    print("Preparing test set ... ", m_test)
    # positive sampling such that train_G is still connected
    test_positive = set()
    while len(test_positive) < m_test:
        i = np.random.randint(low=0, high=len(selected_edges))
        e = selected_edges[i]
        if e not in test_positive:
            G_train.remove_edge(e[0], e[1])
            if nx.is_connected(G_train):
                test_positive.add(e)
            else:
                G_train.add_edge(e[0],e[1])

    test_negative = set()
    while len(test_negative) < m_test:
        i, j = np.random.randint(low=0, high=len(nodes_0)), np.random.randint(low=0, high=len(nodes_1))
        e = (nodes_0[i], nodes_1[j]) if nodes_0[i] < nodes_1[j] else (nodes_1[j], nodes_0[i])
        if nodes_0[i] != nodes_1[j] and not G_edge_type.has_edge(e[0], e[1]):
            test_negative.add(e)

    print("Preparing validation set ... ", m_val)
    val_positive = set()
    while len(val_positive) < m_val:
        i = np.random.randint(low=0, high=len(selected_edges))
        e = selected_edges[i]
        if e not in test_positive and e not in val_positive:
            G_train.remove_edge(e[0], e[1])
            if nx.is_connected(G_train):
                val_positive.add(e)
            else:
                G_train.add_edge(e[0],e[1])

    val_negative = set()
    while len(val_negative) < m_val:
        i, j = np.random.randint(low=0, high=len(nodes_0)), np.random.randint(low=0, high=len(nodes_1))
        e = (nodes_0[i], nodes_1[j]) if nodes_0[i] < nodes_1[j] else (nodes_1[j], nodes_0[i])
        if nodes_0[i] != nodes_1[j] and not G_edge_type.has_edge(e[0], e[1]) and e not in test_negative :
            val_negative.add(e)

    print("Preparing train edges ... ")
    train_edges = []
    if edge_type[0] != edge_type[1]:
        for n0 in nodes_0:
            for n1 in nodes_1:
                e = (n0, n1) if n0 < n1 else (n1, n0)
                train_edges.append(e)
    else:
        for i in range(len(nodes_0)):
            for j in range(i+1, len(nodes_0)):
                e = (nodes_0[i], nodes_0[j]) if nodes_0[i] < nodes_0[j] else (nodes_0[j], nodes_0[i])
                train_edges.append(e)

    train_edges = set(train_edges).difference(test_positive).difference(test_negative)
    print("time:", time.time() - t, "| train edges", len(train_edges))

    print("Is G_train connected?", nx.is_connected(G_train))
    # TODO: remove "dependencies" in some graphs

    return G_train, list(test_positive), list(test_negative), list(val_positive), list(val_negative), list(train_edges)



if __name__ == '__main__':
    # preprocess_dlbp_data(["data/dblp/dblp-ref-0.json", "data/dblp/dblp-ref-1.json", "data/dblp/dblp-ref-2.json", "data/dblp/dblp-ref-3.json"])
    # visualize("data/exp/embed.women.wew.w50.l5.txt", "data/exp/women_tsne.pdf")
    G, A = load_data("data/bio/parsed/bio_edgelist.tsv")
    print(A.shape)