from __future__ import division
from __future__ import print_function

import time
import os


import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
import networkx as nx
import sys

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
# Train on GPU
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

sys.path.insert(0, "..../lpnh")

from utils import read_split, load_graph_data

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
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
dataset_path = "data/" + dataset_str + "/parsed/"
random_seed = FLAGS.random_seed
graph_path = dataset_path + dataset_str + "_edgelist.tsv"
edge_type = FLAGS.edge_type


# Load data
# adj, features = load_data(dataset_str)
G = load_graph_data(graph_path)

# Store original adjacency matrix (without diagonal entries) for later
nodes = list(G.nodes())
print(len(nodes))
adj_orig = nx.to_scipy_sparse_matrix(G, nodelist=nodes)
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

p = dataset_path + "random_splits/" + edge_type + "/random" + str(random_seed) + "/"
print(p)
G_train, test_positive_e, test_negative_e, val_positive_e, val_negative_e, train_edges = read_split(G, edge_type.split("_"), random_seed, p)

t0 = time.time()

adj_train = nx.to_scipy_sparse_matrix(G_train, nodelist=nodes)
adj = adj_train
val_positive = [(nodes.index(e1), nodes.index(e2)) for (e1, e2) in val_positive_e]
val_negative = [(nodes.index(e1), nodes.index(e2)) for (e1, e2) in val_negative_e]
test_positive = [(nodes.index(e1), nodes.index(e2)) for (e1, e2) in test_positive_e]
test_negative = [(nodes.index(e1), nodes.index(e2)) for (e1, e2) in test_negative_e]

if FLAGS.features == 0:
    features = sp.identity(G_train.number_of_nodes())  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, adj_rec=None):
    if adj_rec is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
        adj_rec = np.dot(emb, emb.T)


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges

    preds = []
    pos = []

    for e in edges_pos:
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
    return adj_rec, roc_score, ap_score


val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

best_val_roc = 0
best_preds = None

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    current_preds, roc_curr, ap_curr = get_roc_score(val_positive, val_negative)

    if roc_curr > best_val_roc:  # save best model
        best_val_roc = roc_curr
        best_preds = current_preds

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(roc_curr),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

print("Time:", time.time() - t0)
_, roc_score, ap_score = get_roc_score(test_positive, test_negative, adj_rec=best_preds)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
