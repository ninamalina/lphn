import networkx as nx
import random
import numpy as np
import time
from subprocess import call
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from generate_meta_paths import MetaPathGeneratorBio
from collections import defaultdict
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
import os
from utils import split_test_train

edge_functions = {
    "hadamard": lambda a, b: a * b,
    "average": lambda a, b: 0.5 * (a + b),
    "l1": lambda a, b: np.abs(a - b),
    "l2": lambda a, b: np.abs(a - b) ** 2,
}


class SimpleClassifier:

    def __init__(self, G_train, train_edges, test_positive, test_negative, path):
        self.train_edges = train_edges
        self.test_edges = test_positive + test_negative

        if os.path.exists(path + "train_edges.npy") :
            print ("Reading files")
            train_edges_2 = np.load(path + "train_edges.npy")
            if np.array_equal(self.train_edges, train_edges_2) :
                self.X_train = np.load(path + "X_train.npy")
                self.Y_train = np.load(path + "Y_train.npy")
                self.X_test = np.load(path + "X_test.npy")
                self.Y_test = np.load(path + "Y_test.npy")
        else:
            print("Preparing data")
            self.prepare_train(G_train, train_edges)
            Y_test = [1 for i in test_positive] + [0 for i in test_negative]
            self.prepare_test(G_train, self.test_edges, Y_test)
            self.save("data/bio/parsed/preprocessed_simple/" + edge_type[0] + "_" + edge_type[1] + "/random" + str(num) + "/")


    def train(self, method, seed=0):
        if method == "RF":
            self.clf = RandomForestClassifier(n_estimators=10)
        elif method == "LASSO":
            self.clf = linear_model.Lasso(alpha = 0.1)
        elif method == "LR":
            self.clf = linear_model.LogisticRegression(class_weight="balanced", random_state=seed)
        elif method == "AB":
            self.clf = AdaBoostClassifier(n_estimators=100)
        elif method == "SGD":
            self.clf = linear_model.SGDClassifier(max_iter=1000)

        self.clf.fit(self.X_train, self.Y_train)

    def predict(self):
        self.Y_pred = self.clf.predict(self.X_test)
        return self.Y_pred

    def evaluate(self, metric="AUC"):
        if metric == "AUC":
            return roc_auc_score(self.Y_test, np.round(self.Y_pred))
        elif metric == "confussion":
            return confusion_matrix(self.Y_test, np.round(self.Y_pred))
        else:
            return("Metric not known!")

    def create_features(self, G_train, edge_bunch):
        i = 0
        X = []
        page_rank = nx.pagerank_scipy(G_train)
        for pair in edge_bunch:
            commmon_neighbors = len(list(nx.common_neighbors(G_train, pair[0], pair[1])))
            jaccard_coefficient = nx.jaccard_coefficient(G_train, [pair]).next()[2]
            adamic_adar = nx.adamic_adar_index(G_train, [pair]).next()[2]
            degree_0 = nx.degree(G_train, pair[0])
            degree_1 = nx.degree(G_train, pair[1])
            prod = degree_0 * degree_1
            page_rank_0 = page_rank[pair[0]]
            page_rank_1 = page_rank[pair[1]]

            # try:
            #     shortest_path = nx.shortest_path_length(G_train, pair[0], pair[1])
            #     reciprocal_shortest_path = 1. / shortest_path
            # except nx.NetworkXNoPath:
            #     reciprocal_shortest_path = 0.

            f = [degree_0,
                 degree_1,
                 prod,
                 commmon_neighbors,
                 jaccard_coefficient,
                 adamic_adar,
                 page_rank_0,
                 page_rank_1,
                 # reciprocal_shortest_path
                 ]

            X.append(f)

            i += 1
            if i%1000000 == 0:
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
        np.save(path + "train_edges", self.train_edges)



class PathEmbeddingClassifier:
    def __init__(self, mpg, G_train, train_edges, test_positive, test_negative, meta_path, file_path, seed):
        self.mpg = mpg
        self.seed = seed
        self.train_edges = train_edges
        self.test_edges = test_positive + test_negative

        if not os.path.exists(file_path + "walks.txt"):
            self.generate_walks(meta_path, file_path + "walks.txt")

        self.generate_embeddings(file_path + "walks.txt", file_path + "embeddings", os.path.exists(file_path + "embeddings.txt"))

        if os.path.exists(file_path + "train_edges.npy"):
            print ("Reading files")
            train_edges_2 = np.load(file_path + "train_edges.npy")
            if np.array_equal(self.train_edges, train_edges_2):
                self.X_train = np.load(file_path + "X_train.npy")
                self.Y_train = np.load(file_path + "Y_train.npy")
                self.X_test = np.load(file_path + "X_test.npy")
                self.Y_test = np.load(file_path + "Y_test.npy")
        else:
            print("Preparing data")
            self.prepare_train(G_train, self.train_edges)
            Y_test = [1 for i in test_positive] + [0 for i in test_negative]
            self.prepare_test(self.test_edges, Y_test)
            self.save(file_path)


    def train(self, method):
        print("Training ... ")
        if method == "RF":
            self.clf = RandomForestClassifier(n_estimators=100)
        elif method == "LR":
            self.clf = linear_model.LogisticRegression(class_weight="balanced", random_state=self.seed)
        self.clf.fit(self.X_train, self.Y_train)

    def predict(self):
        print("Predicting ...")
        self.Y_pred = self.clf.predict(self.X_test)
        return self.Y_pred

    def evaluate(self, metric="AUC"):
        if metric == "AUC":
            return roc_auc_score(self.Y_test, self.Y_pred)
        elif metric == "confussion":
            return confusion_matrix(self.Y_test, self.Y_pred)
        else:
            return("Metric not known!")

    def prepare_train(self, G_train, train_edges):
        self.X_train = self.create_features(train_edges)
        self.Y_train = [int(G_train.has_edge(i,j)) for (i,j) in train_edges]

    def prepare_test(self, test_edges, Y_test):
        self.X_test = self.create_features(test_edges)
        self.Y_test = Y_test


    def generate_walks(self, path, out_file):
        print("Generating walks for path " + path)
        if path == "disease-gene-disease":
            self.mpg.generate_random_di_g_di(out_file, 100, 100)
        elif path == "gene-disease-gene":
            self.mpg.generate_random_g_di_g(out_file, 100, 100)
        elif path == "drug-gene-drug":
            self.mpg.generate_random_dr_g_dr(out_file, 100, 100)
        elif path == "gene-drug-gene":
            self.mpg.generate_random_dr_g_dr(out_file, 100, 100)
        else:
            self.mpg.generate_walks(out_file, 10, 10)

    def generate_embeddings(self, walks_file, embed_file, generated):
        if not generated:
            print("Generating embeddings")
            call(["./code_metapath2vec/metapath2vec", "-train", walks_file, "-output", embed_file,
                  "-pp", "1", "-size", "32", "-window", "7", "-debug", "2", "-negative", "5", "-threads", "32"])

        print("Embeddings generated - reading to array")
        f = open(embed_file + ".txt")
        f.readline()
        f.readline()
        self.embeddings = defaultdict(list)
        for line in f:
            toks = line.strip().split(" ")
            self.embeddings[toks[0]] = np.array([float(n) for n in toks[1:]])

        print("Embeddings generated")


    def create_features(self, train_edges, edge_function="hadamard"):
        print("Creating features ... ")
        X = []

        for pair in train_edges:
            # print(pair)
            features = edge_functions[edge_function](self.embeddings[pair[0]], self.embeddings[pair[1]])
            X.append(features)

        return X

    def save(self, path):
        np.save(path + "X_train", self.X_train)
        np.save(path + "X_test", self.X_test)
        np.save(path + "Y_train", self.Y_train)
        np.save(path + "Y_test", self.Y_test)
        np.save(path + "train_edges", self.train_edges)


def read_data(GC, edge_type, num, path=None):

    if path == None:
        return split_test_train(GC, edge_type, num)

    path_G = path + "G_train.edgelist"

    if os.path.exists(path_G):
        G_train = nx.read_edgelist(path_G)
        test_positive = np.load(path + "test_positive.npy")
        test_negative = np.load(path + "test_negative.npy")
        val_positive = np.load(path + "val_positive.npy")
        val_negative = np.load(path + "val_negative.npy")
        train_edges = np.load(path + "train_edges.npy")
    else:
        G_train, test_positive, test_negative, val_positive, val_negative, train_edges = split_test_train(GC, edge_type,
                                                                                                          num)
        nx.write_edgelist(G_train, path_G)
        np.save(path + "val_negative", val_negative)
        np.save(path + "val_positive", val_positive)
        np.save(path + "test_negative", test_negative)
        np.save(path + "test_positive", test_positive)
        np.save(path + "train_edges", train_edges)


    return G_train, test_positive, test_negative, val_positive, val_negative, train_edges



if __name__ == '__main__':

    f = open("data/bio/parsed/bio_edgelist.tsv")
    G = nx.Graph()
    i = 0

    for line in f:
        a, b = line.strip().split("\t")
        G.add_edge(a, b)

    print("Created graph")

    edge_types = [("disease", "gene"), ("drug","gene"), ("gene", "gene")]

    G.remove_edges_from(G.selfloop_edges())
    GC = max(nx.connected_component_subgraphs(G), key=len) # take greatest connected component

    for edge_type in edge_types:
        for num in range(1):

            print(num, edge_type)
            p = "data/bio/parsed/random_splits/" + edge_type[0] + "_" + edge_type[1] + "/random" + str(num) + "/"
            G_train, test_positive, test_negative, val_positive, val_negative, train_edges = read_data(GC, edge_type, num, p)

            print("Test train split")
            #
            #
            # print("Simple classifier")
            # file_path = "data/bio/parsed/preprocessed_simple/" + edge_type[0] + "_" + edge_type[1] + "/random" + str(num) + "/"
            # simple_model = SimpleClassifier(G_train, train_edges, test_positive, test_negative, file_path)
            # simple_model.train("LR", num)
            # simple_model.predict()
            # print("AUC:", simple_model.evaluate())
            # print("confussion:", simple_model.evaluate(metric="confussion"))
            #
            #
            #
            # ### path embedding classifier
            # print("Meta path classifier")
            # mpg = MetaPathGeneratorBio(num)
            # mpg.read_data(G_train)
            # # if edge_type == ("disease", "gene"):
            # #     meta_path = "gene-disease-gene"
            # # elif edge_type == ("drug", "gene"):
            # #     meta_path = "gene-drug-gene"
            # # elif edge_type == ("gene","gene"):
            # #     meta_path = "gene-disease-gene"
            #
            # meta_path = "disease_gene"
            #
            # file_path ="data/bio/parsed/embeddings/" + edge_type[0] + "_" + edge_type[1] + "/random" + str(num) + "/"
            # metapath_model = PathEmbeddingClassifier(mpg, G_train, train_edges, test_positive, test_negative, meta_path, file_path, num)
            #
            # metapath_model.train("LR")
            # metapath_model.predict()
            # print("AUC:", metapath_model.evaluate())
            # print("confussion:", metapath_model.evaluate(metric="confussion"))
            #
            #
            # # gcn
            #
            # adj_train = nx.to_scipy_sparse_matrix(G_train)