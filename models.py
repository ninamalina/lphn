import networkx as nx
from itertools import combinations
import random
from itertools import product
import numpy as np
import time
import os.path
from subprocess import call
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}

def split_test_train(G, edge_type, test_size=0.2):
    nodes_0 = [n for n in G.nodes if n.startswith(edge_type[0])]
    nodes_1 = [n for n in G.nodes if n.startswith(edge_type[1])]

    selected_edges = [e for e in G.edges if (e[0].startswith(edge_type[0]) and e[1].startswith(edge_type[1]))]

    G_edge_type = nx.Graph()
    G_edge_type.add_nodes_from(nodes_0 + nodes_1)
    G_edge_type.add_edges_from(selected_edges)

    n_edges = G_edge_type.number_of_edges()
    m = int(test_size * n_edges)
    test_positive = random.sample(G_edge_type.edges, m)
    test_positive = [(e[1], e[0]) if e[0] > e[1] else e for e in test_positive]
    # print(test_positive)

    test_negative = set()

    while len(test_negative) < m:
        i, j = np.random.randint(low=0, high=len(nodes_0)), np.random.randint(low=0, high=len(nodes_1))
        if not G_edge_type.has_edge(nodes_0[i], nodes_1[j]):
            e = (nodes_0[i], nodes_1[j]) if nodes_0[i] < nodes_1[j] else (nodes_1[j], nodes_0[i])
            test_negative.add(e)

    # print(test_negative)

    print(len(nodes_0), len(nodes_1))
    print(len(test_positive), len(test_negative))

    t = time.time()
    train_edges = []
    ind = 0
    if edge_type[0] != edge_type[1]:
        for n0 in nodes_0:
            for n1 in nodes_1:
                e = (n0, n1) if n0 < n1 else (n1, n0)
                train_edges.append(e)
                ind += 1
                if ind % 10000000 == 0:
                    print(ind)
    else:
        for i in range(len(nodes_0)):
            for j in range(i, len(nodes_0)):
                e = (nodes_0[i], nodes_1[j]) if nodes_0[i] < nodes_1[j] else (nodes_1[j], nodes_0[i])
                train_edges.append(e)
                ind += 1
                if ind % 10000000 == 0:
                    print(ind)


    print(len(train_edges))

    train_edges = set(train_edges).difference(test_positive).difference(test_negative)
    print("time:", time.time() - t, "| train edges", len(train_edges))
    G_train = G.copy()
    G_train.remove_edges_from(test_positive)
    # TODO: remove "dependencies" in some graphs

    print(len(test_positive), len(test_negative), len(train_edges))
    print(len(test_positive) + len(test_negative) + len(train_edges))
    return G, G_train, list(test_positive), list(test_negative), list(train_edges)


class SimpleClassifier:
    def __init__(self, G, seed, method="RF"):
        self.G = G
        self.method = method
        random.seed(seed)
        np.random.seed(seed)

    def train(self):
        if self.method == "RF":
            # print(self.X_train)
            # print(self.Y_train)
            self.clf = RandomForestClassifier(n_estimators=10)
            self.clf.fit(self.X_train, self.Y_train)

        # linear regression; svm; random forest

    def predict(self):
        self.Y_pred = self.clf.predict(self.X_test)
        return self.Y_pred

    def evaluate(self, metric="AUC"):
        if metric == "AUC":
            return roc_auc_score(self.Y_test, self.Y_pred)
        elif metric == "confussion":
            return confusion_matrix(self.Y_test, self.Y_pred)
        else:
            return("Metric not known!")

    def create_features(self, G_train, edge_bunch):
        i = 0
        X = []
        t = time.time()
        page_rank = nx.pagerank_scipy(G_train)
        t1 = time.time()
        print("pagerank time", t1 - t)
        for pair in edge_bunch:
            commmon_neighbors = len(list(nx.common_neighbors(G_train, pair[0], pair[1])))
            jaccard_coefficient = nx.jaccard_coefficient(G_train, [pair]).next()[2]
            adamic_adar = nx.adamic_adar_index(G_train, [pair]).next()[2]
            degree_0 = nx.degree(G_train, pair[0])
            degree_1 = nx.degree(G_train, pair[1])
            prod = degree_0 * degree_1
            page_rank_0 = page_rank[pair[0]]
            page_rank_1 = page_rank[pair[1]]

            try:
                shortest_path = nx.shortest_path_length(G_train, pair[0], pair[1])
                reciprocal_shortest_path = 1 / shortest_path
            except nx.NetworkXNoPath:
                reciprocal_shortest_path = 0

            f = [degree_0,
                 degree_1,
                 prod,
                 commmon_neighbors,
                 jaccard_coefficient,
                 adamic_adar,
                 page_rank_0,
                 page_rank_1,
                 reciprocal_shortest_path]

            X.append(f)

            i += 1
            if i%100000 == 0:
                print(i)

        return X

    def prepare_train(self, G_train, train_edges):
        self.X_train = self.create_features(G_train, train_edges)
        self.Y_train = [int(G_train.has_edge(i,j)) for (i,j) in train_edges]

    def prepare_test(self, G_train, test_edges, Y_test):
        self.X_test = self.create_features(G_train, test_edges)
        self.Y_test = Y_test

    def save(self, path):
        np.save(path + "X_train", self.X_train)
        np.save(path + "X_test", self.X_test)
        np.save(path + "Y_train", self.Y_train)
        np.save(path + "Y_test", self.Y_test)


class PathEmbeddingClassifier:
    def __init__(self, G):
        self.G = G
        pass

    def evaluate(self):
        pass

    def generate_paths(self, path, out_file):
        if os.path.isfile(out_file):
            pass
        else:
            #TODO: generate file with paths
            pass

    def generate_embeddings(self, out_file=None):
        if out_file is not None: #os.path.isfile(out_file):
            pass
        else:
            #TODO: generate file with paths
            call(["./code_metapath2vec/metapath2vec", "-train", "data/exp/output.women.w50.l5.txt", "-output", "data/exp/embed.women.wew.w50.l5",
                  "-pp", "1", "-size", "32", "-window", "7", "-negative", "5", "-threads", "32"])
            # ./metapath2vec -train ../in_dbis/dbis.cac.w1000.l100.txt -output ../out_dbis/dbis.cac.w1000.l100 -pp 1 -size 128 -window 7 -negative 5 -threads 32

            pass


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


    for edge_type in edge_types:
        for num in range(5):
            print(num, edge_type)
            G, G_train, test_positive, test_negative, train_edges = split_test_train(GC, edge_type)
            simple_model = SimpleClassifier(GC, num)
            simple_model.prepare_train(G_train, train_edges)
            Y_test = [1 for i in test_positive] + [0 for i in test_negative]
            simple_model.prepare_test(G_train, test_positive + test_negative, Y_test)

            simple_model.save("data/bio/parsed/preprocessed_simple/" + edge_type[0] + "_" + edge_type[1] + "/random" + str(num) + "/")
            simple_model.train()
            simple_model.predict()
            print("AUC:", simple_model.evaluate())
            print("confussion:", simple_model.evaluate(metric="confussion"))
