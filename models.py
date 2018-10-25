import networkx as nx
from itertools import combinations
import random
from itertools import product
# from numpy import random
import numpy as np
import time

class Simple():
    def __init__(self, G, edge_types):
        self.G = G
        self.edge_types = edge_types
        pass

    def train(self):
        # linear regression; svm; random forest
        pass

    def predict(self):
        pass


    def evaluate(self):
        for edge_type in self.edge_types:
            print (edge_type)
            nodes_0 = [n for n in self.G.nodes if n.startswith(edge_type[0])]
            nodes_1 = [n for n in self.G.nodes if n.startswith(edge_type[1])]
            selected_edges = [e for e in self.G.edges if (e[0].startswith(edge_type[0]) and e[1].startswith(edge_type[1]))]
            G_edge_type = nx.Graph()
            G_edge_type.add_nodes_from(nodes_0 + nodes_1)
            G_edge_type.add_edges_from(selected_edges)

            n_edges = G_edge_type.number_of_edges()
            m = int(0.2 * n_edges)
            sampled_positive = random.sample(G_edge_type.edges, m)

            sampled_negative = set()

            # indices_0 = np.random.randint(low=0, high=len(nodes_0), size=2*m)
            # indices_1 = np.random.randint(low=0, high=len(nodes_1), size=2*m)
            # indices = zip(indices_0, indices_1)

            while len(sampled_negative) < m:
                i, j = np.random.randint(low=0, high=len(nodes_0)), np.random.randint(low=0, high=len(nodes_1))
                # i, j = indices.pop()
                if not G_edge_type.has_edge(nodes_0[i], nodes_1[j]):
                    sampled_negative.add((nodes_0[i], nodes_1[j]))

            print(4)

            print(len(sampled_positive))
            print(len(sampled_negative))

            print(5)
            t = time.time()
            all_possible_edges = []
            ind = 0
            if edge_type[0] != edge_type[1]:
                for n0 in nodes_0:
                    for n1 in nodes_1:
                        all_possible_edges.append((n0, n1))
                        ind += 1
                        if ind % 10000000 == 0:
                            print(ind)
            else:
                for i in range(len(nodes_0)):
                    for j in range(i, len(nodes_0)):
                        all_possible_edges.append((nodes_0[i], nodes_0[j]))
                        ind += 1
                        if ind % 10000000 == 0:
                            print(ind)

            print(time.time() - t)

            G_train = self.G.copy()
            G_train.remove_edges_from(sampled_positive)
            # TODO: remove "dependencies" in some graphs
            t = time.time()
            X = self.create_features(G_train, all_possible_edges)
            print("time to create features:", time.time() - t)
            # Y = [int(self.G.has_edge(pair[0], pair[1])) for pair in edge_bunch]
            # train

            break


    def create_features(self, G_train, edge_bunch):
        i = 0
        X = []
        t = time.time()
        page_rank = nx.pagerank_scipy(G_train)
        t1 = time.time()
        print(t1 - t)
        shortest_paths = dict(nx.all_pairs_shortest_path_length(G_train, cutoff=10))
        t2 = time.time()
        print(t2 - t1)
        for pair in edge_bunch:
            commmon_neighbors = nx.common_neighbors(G_train, pair[0], pair[1])
            jaccard_coefficient = nx.jaccard_coefficient(G_train, [pair])
            adamic_adar = nx.adamic_adar_index(G_train, [pair])
            degree_0 = nx.degree(G_train, pair[0])
            degree_1 = nx.degree(G_train, pair[1])
            prod = degree_0 * degree_1
            page_rank_0 = page_rank[pair[0]]
            page_rank_1 = page_rank[pair[1]]

            if pair[0] in shortest_paths and pair[1] in shortest_paths[pair[0]]:
                reciprocal_shortest_path = 1 / shortest_paths[pair[0]][pair[1]] # preveri ce obstaja, ce ne, 0
            else:
                reciprocal_shortest_path = 0

            f = [degree_0,
                 degree_1,
                 prod,
                 commmon_neighbors,
                 jaccard_coefficient,
                 adamic_adar,
                 page_rank_0,
                 page_rank_1,
                 reciprocal_shortest_path
            ]
            X.append(f)

            i += 1
            if i%100000 == 0:
                print(i)

        return X

if __name__ == '__main__':

    f = open("data/bio/parsed/bio_edgelist.tsv")
    G = nx.Graph()
    i = 0

    for line in f:
        a, b = line.strip().split("\t")
        G.add_edge(a, b)

    print("Created graph")

    # genes = [n for n in G.nodes if n.startswith("gene")]
    # diseases = [n for n in G.nodes if n.startswith("disease")]
    # drugs = [n for n in G.nodes if n.startswith("drug")]

    edge_types = [("disease", "gene"), ("drug","gene"), ("gene", "gene")]

    GC = max(nx.connected_component_subgraphs(G), key=len) # take greatest connected component
    simple_model = Simple(GC, edge_types)
    simple_model.evaluate()
