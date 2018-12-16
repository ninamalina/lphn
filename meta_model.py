import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import os
import sys
from utils import read_split
from generate_meta_paths import MetaPathGeneratorBio
from subprocess import call
from collections import defaultdict


# edge_functions = {
#     "hadamard": lambda a, b: a * b,
#     "average": lambda a, b: 0.5 * (a + b),
#     "l1": lambda a, b: np.abs(a - b),
#     "l2": lambda a, b: np.abs(a - b) ** 2,
# }

class PathEmbeddingClassifier:
    def __init__(self, mpg, G_train, train_edges, test_positive, test_negative, meta_path, file_path, seed, numwalks, length):
        self.mpg = mpg
        self.seed = seed
        self.numwalks = numwalks
        self.length = length
        self.train_edges = train_edges
        self.test_edges = np.concatenate((test_positive, test_negative), axis=0)
        self.generate_walks(meta_path, file_path + meta_path + "_walks.txt")
        self.generate_embeddings(file_path + meta_path + "_walks.txt", file_path + meta_path + "_embeddings", False)
        self.prepare_train(G_train, self.train_edges)
        Y_test = [1 for i in test_positive] + [0 for i in test_negative]
        self.prepare_test(self.test_edges, Y_test)


    def train(self, method):
        print("Training ... ")
        if method == "RF":
            self.clf = RandomForestClassifier(n_estimators=100)
        elif method == "LR":
            self.clf = linear_model.LogisticRegression(class_weight="balanced", random_state=self.seed)
        self.clf.fit(self.X_train, self.Y_train)

    def predict(self, prob = False):
        print("Predicting ...")
        if prob:
            self.Y_pred = self.clf.predict_proba(self.X_test)[:,1]
        else:
            self.Y_pred = self.clf.predict(self.X_test)
        return self.Y_pred

    def evaluate(self, metric="AUC"):
        if metric == "AUC":
            return roc_auc_score(self.Y_test, self.Y_pred)
        elif metric == "confussion":
            return confusion_matrix(self.Y_test, np.round(self.Y_pred))
        else:
            return("Metric not known!")

    def prepare_train(self, G_train, train_edges):
        self.X_train = self.create_features(train_edges)
        self.Y_train = [int(G_train.has_edge(i,j)) for (i,j) in train_edges]

    def prepare_test(self, test_edges, Y_test):
        self.X_test = self.create_features(test_edges)
        self.Y_test = Y_test


    def generate_walks(self, meta_path, out_file):
        print("Generating walks for path " + meta_path)

        if meta_path == "all_combined":
            self.mpg.generate_walks(out_file, self.numwalks, self.length)
        elif meta_path == "long":
            self.mpg.generate_walks_2(out_file, self.numwalks, self.length)

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
            features = self.embeddings[pair[0]] * self.embeddings[pair[1]]
            # features = edge_functions[edge_function](self.embeddings[pair[0]], self.embeddings[pair[1]])
            X.append(features)
        return X

    def save(self, path):
        np.save(path + "X_train", self.X_train)
        np.save(path + "X_test", self.X_test)
        np.save(path + "Y_train", self.Y_train)
        np.save(path + "Y_test", self.Y_test)
        np.save(path + "train_edges", self.train_edges)



if __name__ == '__main__':

    # python meta_model.py data/bio/parsed/ data/bio/parsed/bio_edgelist.tsv disease_gene 0
    path = sys.argv[1] # "data/bio/parsed/"
    graph_path = sys.argv[2] # "data/bio/parsed/bio_edgelist.tsv"
    edge_type = sys.argv[3].split("_") # disease_gene
    num = int(sys.argv[4]) # 0
    meta_path = sys.argv[5] # all_combined OR long
    numwalks = int(sys.argv[6]) # 100
    length = int(sys.argv[7]) # 100


    f = open(graph_path)
    G = nx.Graph()

    for line in f:
        a, b = line.strip().split("\t")
        G.add_edge(a, b)

    G.remove_edges_from(G.selfloop_edges())
    GC = max(nx.connected_component_subgraphs(G), key=len) # take greatest connected component

    p = path + "random_splits/" + edge_type[0] + "_" + edge_type[1] + "/random" + str(num) + "/"
    G_train, test_positive, test_negative, val_positive, val_negative, train_edges = read_split(GC, edge_type, num, p)


    print("Meta path classifier")
    mpg = MetaPathGeneratorBio(num)
    mpg.read_data(G_train)


    file_path = path + "embeddings/" + edge_type[0] + "_" + edge_type[1] + "/random" + str(num) + "/"
    metapath_model = PathEmbeddingClassifier(mpg, G_train, train_edges, test_positive, test_negative, meta_path, file_path, num, numwalks, length)

    metapath_model.train("LR")
    metapath_model.predict()
    print("Acc:", metapath_model.evaluate())
    metapath_model.predict(prob=True)
    print("AUC:", metapath_model.evaluate())

    print("confussion:", metapath_model.evaluate(metric="confussion"))