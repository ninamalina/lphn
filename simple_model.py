import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
import os
import sys
from utils import read_split


class SimpleClassifier:

    def __init__(self, G_train, train_edges, test_positive, test_negative, val_positive, val_negative, path):
        self.train_edges = train_edges
        self.test_edges = np.concatenate((test_positive, test_negative), axis=0)
        self.val_edges = np.concatenate((val_positive, val_negative), axis=0)

        if os.path.exists(path + "train_edges.npy") :
            print ("Reading files")
            train_edges_2 = np.load(path + "train_edges.npy")
            if np.array_equal(self.train_edges, train_edges_2) :
                self.X_train = np.load(path + "X_train.npy")
                self.Y_train = np.load(path + "Y_train.npy")
                self.X_test = np.load(path + "X_test.npy")
                self.Y_test = np.load(path + "Y_test.npy")
                self.X_val = np.load(path + "X_val.npy")
                self.Y_val = np.load(path + "Y_val.npy")
            else:
                print("Something is wrong ... ")
        else:
            print("Preparing data")
            self.prepare_train(G_train, train_edges)
            Y_test = [1 for i in test_positive] + [0 for i in test_negative]
            self.prepare_test(G_train, self.test_edges, Y_test)
            Y_val = [1 for i in val_positive] + [0 for i in val_negative]
            self.prepare_val(G_train, self.val_edges, Y_val)
            self.save(path)


    def train(self, method, seed=0):
        if method == "LR":
            self.clf = linear_model.LogisticRegression(class_weight="balanced", random_state=seed)
            self.clf.fit(self.X_train, self.Y_train)
        elif method == "FS": # feature selection
            features = {"degrees" : [0, 1], "prod": [2], "neighbors":[3], "jaccard":[4], "adamic":[5], "pagerank":[6,7]}
            best_features = []
            best_roc = 0
            is_improving = True
            while is_improving:
                is_improving = False
                for feature in features:
                    current_selected = best_features + features[feature]
                    print(current_selected)
                    clf = linear_model.LogisticRegression(class_weight="balanced", random_state=seed)
                    clf.fit(self.X_train[:,current_selected], self.Y_train)
                    pred = clf.predict(self.X_val[:,current_selected])
                    current_roc = roc_auc_score(self.Y_val, pred)
                    if current_roc > best_roc:
                        is_improving = True
                        feature_to_add = feature
                if is_improving:
                    best_features = best_features + features[feature_to_add]
                    features = {feature:features[feature] for feature in features if feature!=feature_to_add}

            self.clf = linear_model.LogisticRegression(class_weight="balanced", random_state=seed)
            self.clf.fit(self.X_train[:,best_features], self.Y_train)
            print(best_features)

    def predict(self, prob=False):
        if prob:
            self.Y_pred = self.clf.predict_proba(self.X_test)[:,1]
            print(self.Y_pred)
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

            f = [degree_0,
                 degree_1,
                 prod,
                 commmon_neighbors,
                 jaccard_coefficient,
                 adamic_adar,
                 page_rank_0,
                 page_rank_1,
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

    def prepare_val(self, G_train, val_edges, Y_val):
        self.X_val = self.create_features(G_train, val_edges)
        self.Y_val = Y_val

    def save(self, path):
        np.save(path + "X_train", self.X_train)
        np.save(path + "X_test", self.X_test)
        np.save(path + "X_val", self.X_val)
        np.save(path + "Y_train", self.Y_train)
        np.save(path + "Y_test", self.Y_test)
        np.save(path + "Y_val", self.Y_val)
        np.save(path + "train_edges", self.train_edges)


if __name__ == '__main__':

    # python simple_model.py data/bio/parsed/ data/bio/parsed/bio_edgelist.tsv disease_gene 0
    path = sys.argv[1] # "data/bio/parsed/"
    graph_path = sys.argv[2] # "data/bio/parsed/bio_edgelist.tsv"
    edge_type = sys.argv[3].split("_") # disease_gene
    num = int(sys.argv[4]) # 0
    method = sys.argv[5] # LR or FS

    f = open(graph_path)
    G = nx.Graph()

    for line in f:
        a, b = line.strip().split("\t")
        G.add_edge(a, b)

    G.remove_edges_from(G.selfloop_edges())
    GC = max(nx.connected_component_subgraphs(G), key=len) # take greatest connected component

    p = path + "random_splits/" + edge_type[0] + "_" + edge_type[1] + "/random" + str(num) + "/"
    G_train, test_positive, test_negative, val_positive, val_negative, train_edges = read_split(GC, edge_type, num, p)

    p = path + "features/" + edge_type[0] + "_" + edge_type[1] + "/random" + str(num) + "/"
    simple_model = SimpleClassifier(G_train, train_edges, test_positive, test_negative, val_positive, val_negative, p)
    simple_model.train(method, num)
    simple_model.predict()
    print("Acc:", simple_model.evaluate())
    simple_model.predict(prob=True)
    print("AUC:", simple_model.evaluate())
    print("confussion:", simple_model.evaluate(metric="confussion"))