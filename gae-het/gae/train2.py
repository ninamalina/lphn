from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp


from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE, DecagonOptimizer
from input_data import load_data
from model import GCNModelAEHet
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
import networkx as nx
from utils import read_split, load_graph_data, check_desc, get_edge_adj_matrices



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'bio', 'Dataset string.')
flags.DEFINE_string('edge_type', 'drug_gene', 'Type of edges to learn.')
flags.DEFINE_integer('random_seed', 0, 'Random seed')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
dataset_path = "../../data/" + dataset_str + "/parsed/"
random_seed = FLAGS.random_seed
graph_path = dataset_path + dataset_str + "_edgelist.tsv"
edge_type = FLAGS.edge_type


edge_types = ["disease_gene", "drug_gene", "gene_gene"]
type_num_dict = {"disease": 0, "drug": 1, "gene":2}
# Load data
G = load_graph_data(graph_path)

adjs_orig = get_edge_adj_matrices(G, {et:None for et in edge_types})


# # get adjajcency matrices for subgraphs
adj_orig = nx.to_scipy_sparse_matrix(G)
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
#
#


p = dataset_path + "random_splits/" + edge_type + "/random" + str(random_seed) + "/"
G_train, test_positive_e, test_negative_e, val_positive_e, val_negative_e, train_edges = read_split(G, edge_type.split("_"), random_seed, p)

adjs_train = get_edge_adj_matrices(G_train, {et: adjs_orig[et]["nodes"] for et in adjs_orig})


adj_train = nx.to_scipy_sparse_matrix(G_train)
# adj = adj_train
nodes = list(G_train.nodes())
val_positive = [(nodes.index(e1), nodes.index(e2)) for (e1, e2) in val_positive_e]
val_negative = [(nodes.index(e1), nodes.index(e2)) for (e1, e2) in val_negative_e]
test_positive = [(nodes.index(e1), nodes.index(e2)) for (e1, e2) in test_positive_e]
test_negative = [(nodes.index(e1), nodes.index(e2)) for (e1, e2) in test_negative_e]
#
print("Dataset read")

if FLAGS.features == 0:
    features = sp.identity(G_train.number_of_nodes())  # featureless

adj = adj_train
#Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'batch': tf.placeholder(tf.int32, name='batch'),
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'edge_type_idx': tf.placeholder(tf.int32, shape=(), name='edge_type_idx'),
    'row_edge_type': tf.placeholder(tf.int32, shape=(), name='row_edge_type'),
    'col_edge_type': tf.placeholder(tf.int32, shape=(), name='col_edge_type'),
}


num_nodes = adj_train.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]

features_nonzero = features[1].shape[0]

print(num_features)

num_feat = {
    0: 0,
    1: 0,
    2: 0,
}
nonzero_feat = {
    0: 0,
    1: 0,
    2: 0
}

# data representation
adj_mats_orig = {
    (1, 2): [adjs_orig["drug_gene"]["adj"]],
    (2, 1): [adjs_orig["drug_gene"]["adj"].transpose(copy=True)],
    (0, 2): [adjs_orig["disease_gene"]["adj"]],
    (2, 0): [adjs_orig["disease_gene"]["adj"].transpose(copy=True)],
    (2, 2): [adjs_orig["gene_gene"]["adj"], adjs_orig["gene_gene"]["adj"].transpose(copy=True)],
}
edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (1, 2): 'bilinear',
    (2, 1): 'bilinear',
    (0, 2): 'bilinear',
    (2, 0): 'bilinear',
    (2, 2): 'bilinear',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())

placeholders.update({'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
                        for i, j in edge_types for k in range(edge_types[i, j])})
placeholders.update({'feat_%d' % i: tf.sparse_placeholder(tf.float32)
                        for i, _ in edge_types})


adj_mats_train = {
    (1, 2): [adjs_train["drug_gene"]["adj"]],
    (2, 1): [adjs_train["drug_gene"]["adj"].transpose(copy=True)],
    (0, 2): [adjs_train["disease_gene"]["adj"]],
    (2, 0): [adjs_train["disease_gene"]["adj"].transpose(copy=True)],
    (2, 2): [adjs_train["gene_gene"]["adj"], adjs_train["gene_gene"]["adj"].transpose(copy=True)],
}

# placeholders.update({'adj_mats_%d,%d,%d' % (i, j, k): sparse_to_tuple(adj_mats_train[i,j][k])
placeholders.update({'adj_mats_%d,%d,%d' % (i, j, k): adj_mats_train[i,j][k]
                     for i, j in edge_types for k in range(edge_types[i, j])})


idx2edge_type = {}
edge_type2idx = {}
r = 0
for i, j in edge_types:
    edge_type2idx[i, j] = r
    idx2edge_type[r] = i, j
    r += 1

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAEHet(placeholders,
                          num_feat,
                          nonzero_feat,
                          edge_types=edge_types,
                          decoders=edge_type2decoder)


pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = DecagonOptimizer(embeddings=model.embeddings,
                                latent_inters=model.latent_inters,
                                latent_varies=model.latent_varies,
                                # degrees=degrees,
                                edge_types=edge_types,
                                edge_type2dim=edge_type2dim,
                                placeholders=placeholders,
                                # batch_size=FLAGS.batch_size,
                                # margin=FLAGS.max_margin)
                                labels = "" #TODO: real labels for edge type
                               )

    # opt = OptimizerAE(preds=model.reconstructions,
    #                   labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
    #                                                               validate_indices=False), [-1]),
    #                   pos_weight=pos_weight,
    #                   norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    # print(adj_rec)
    for e in edges_pos:
        # print(e)

        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()

    edge_types_iterator = list(range(num_edge_types))

    for et in edge_types_iterator:

        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders, et, idx2edge_type)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        roc_curr, ap_curr = get_roc_score(val_positive, val_negative)
        val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

roc_score, ap_score = get_roc_score(test_positive, test_negative)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
