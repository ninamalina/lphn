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

from gaehet.optimizer import OptimizerAE
from gaehet.model import GCNModelAEHet
from gaehet.preprocessing import construct_feed_dict, sparse_to_tuple, preprocess_graph
import networkx as nx
from utils import read_split, load_graph_data, get_edge_adj_matrices

def construct_placeholders(edge_types):
    placeholders = {
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({'adj_mats_%d,%d' % (i, j): tf.sparse_placeholder(tf.float32) for i, j in edge_types})
    placeholders.update({'feat_%d' % i: tf.sparse_placeholder(tf.float32) for i, _ in edge_types})
    return placeholders

def get_roc_score(edges_pos, edges_neg, edge_type, emb=None, adj_rec=None):
    if adj_rec is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.reconstructions, feed_dict=feed_dict)
        adj_rec = emb[edge_type]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_mats_orig[edge_type][e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_mats_orig[edge_type][e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    print(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    print(labels_all)
    print(np.hstack([pos, neg]))
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return adj_rec, roc_score, ap_score


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

# flags.DEFINE_string('model', 'model_ae', 'Model string.')
flags.DEFINE_string('optimizer', 'multi', 'Model string.')
flags.DEFINE_string('decoder', 'innerproduct', 'Model string.')
flags.DEFINE_string('dataset', 'bio', 'Dataset string.')
flags.DEFINE_string('edge_type', 'drug_gene', 'Type of edges to learn.')
flags.DEFINE_integer('random_seed', 0, 'Random seed')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')

# model_str = FLAGS.model
optimizer_str = FLAGS.optimizer
decoder_str = FLAGS.decoder
dataset_str = FLAGS.dataset
dataset_path = "data/" + dataset_str + "/parsed/"
random_seed = FLAGS.random_seed
graph_path = dataset_path + dataset_str + "_edgelist.tsv"
edge_type = FLAGS.edge_type


edge_types_strings = ["disease_gene", "drug_gene", "gene_gene"]
type_num_dict = {"disease": 0, "drug": 1, "gene":2}
num_type_dict = {0: "disease", 1: "drug", 2: "gene"}

edge_types = []
for et in edge_types_strings:
    (i,j) = et.split("_")
    edge_types.append((type_num_dict[i],type_num_dict[j]))
    if i != j:
        edge_types.append((type_num_dict[j],type_num_dict[i]))

# Load data
G = load_graph_data(graph_path)

adjs_orig = get_edge_adj_matrices(G, {et:None for et in edge_types_strings})

# # get adjajcency matrices for subgraphs
adj_orig = nx.to_scipy_sparse_matrix(G)
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

p = dataset_path + "random_splits/" + edge_type + "/random" + str(random_seed) + "/"
G_train, test_positive_e, test_negative_e, val_positive_e, val_negative_e, train_edges = read_split(G, edge_type.split("_"), random_seed, p)

adjs_train = get_edge_adj_matrices(G_train, {et: adjs_orig[et]["nodes"] for et in adjs_orig})

adj_train = nx.to_scipy_sparse_matrix(G_train)
# adj = adj_train

k = tuple([type_num_dict[t] for t in edge_type.split("_")])
print("k", k, edge_type)

nodes0 = adjs_orig[edge_type]["nodes"][0]
nodes1 = adjs_orig[edge_type]["nodes"][1]

val_positive = [(nodes0.index(e1), nodes1.index(e2)) for (e1, e2) in val_positive_e]
val_negative = [(nodes0.index(e1), nodes1.index(e2)) for (e1, e2) in val_negative_e]
test_positive = [(nodes0.index(e1), nodes1.index(e2)) for (e1, e2) in test_positive_e]
test_negative = [(nodes0.index(e1), nodes1.index(e2)) for (e1, e2) in test_negative_e]

print("Dataset read")

adj_mats_orig = {}
adj_mats_train = {}

for (i,j) in edge_types:
    type_i, type_j = num_type_dict[i], num_type_dict[j]

    if type_i + "_" + type_j in adjs_orig:
        adj_mats_orig[i,j] = adjs_orig[type_i + "_" + type_j]["adj"]
        adj_mats_train[i,j] = adjs_train[type_i + "_" + type_j]["adj"]
    else:
        adj_mats_orig[i, j] = adjs_orig[type_j + "_" + type_i]["adj"].transpose(copy=True)
        adj_mats_train[i, j] = adjs_train[type_j + "_" + type_i]["adj"].transpose(copy=True)

adj_mats_train_normalized = {et: preprocess_graph(adj_mats_train[et]) for et in edge_types}

for i,j in adj_mats_train:
    print(i, j, adj_mats_train[i,j].shape)

idx2edge_type = {}
r = 0
for i, j in edge_types:
    idx2edge_type[r] = i, j
    r += 1

obj_type2n = {i: adj_mats_train[i,j].shape[0] for i, j in edge_types}
print(obj_type2n)

if FLAGS.features == 0:
    features = { o: sparse_to_tuple(sp.identity(obj_type2n[o]).tocoo()) for o in obj_type2n }
    num_feat = { i: features[i][2][1] for i in features}
    nonzero_feat = { i: features[i][1].shape[0] for i in features}

    print("numfeat", num_feat)
    print("nonzero_feat", nonzero_feat)

edge_type2decoder = { et: decoder_str for et in edge_types }

placeholders = construct_placeholders(edge_types)

# Create model
model = GCNModelAEHet(placeholders,
                      num_feat,
                      nonzero_feat,
                      edge_types=edge_types,
                      decoders=edge_type2decoder)

pos_weights = {et: float(adj_mats_train[et].shape[0] * adj_mats_train[et].shape[0] - adj_mats_train[et].sum()) / adj_mats_train[et].sum() for et in edge_types}
norms = {et: adj_mats_train[et].shape[0] * adj_mats_train[et].shape[0] / float((adj_mats_train[et].shape[0] * adj_mats_train[et].shape[0] - adj_mats_train[et].sum()) * 2) for et in edge_types}

# Optimizer
with tf.name_scope('optimizer'):
    if optimizer_str == 'single':
        opt = OptimizerAE(
            preds = model.reconstructions[k],
            labels = model.adj_mats[k],
            norm = norms[k],
            pos_weight = pos_weights[k],
        )
    elif optimizer_str == "multi":
        optimizers = []
        for i in range(len(edge_types)):
            optimizers.append(
                OptimizerAE(
                    preds=model.reconstructions[idx2edge_type[i]],
                    labels=model.adj_mats[idx2edge_type[i]],
                    norm=norms[idx2edge_type[i]],
                    pos_weight=pos_weights[idx2edge_type[i]],
                )
            )

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []
val_roc_score = []

# adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = sparse_to_tuple(adj_label)

best_val_roc = 0
best_preds = None

# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()

    for idx in idx2edge_type:
        # print (idx)

        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_mats_train, adj_mats_train_normalized, features, edge_types, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        if optimizer_str == "multi":
            outs = sess.run([optimizers[idx].opt_op, optimizers[idx].cost], feed_dict=feed_dict)
        else:
            outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]

        current_preds, roc_curr, ap_curr = get_roc_score(val_positive, val_negative, k)

        if roc_curr > best_val_roc: # save best model
            best_val_roc = roc_curr
            best_preds = current_preds

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
      "train_acc=", "val_roc=", "{:.5f}".format(roc_curr),
      "val_ap=", "{:.5f}".format(ap_curr),
      "time=", "{:.5f}".format(time.time() - t))

        # val_roc_score.appenxd(roc_curr)

    # if len(val_roc_score) > 100 and check_desc(val_roc_score[-4:]): # stop learning if roc dropping
    #     break

print("Optimization Finished!")

_, roc_score, ap_score = get_roc_score(test_positive, test_negative, k, adj_rec=best_preds)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
