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
# from input_data import load_data
from gaehet.model import GCNModelAEHet
from gaehet.preprocessing import construct_feed_dict, sparse_to_tuple, mask_test_edges, preprocess_graph
import networkx as nx
from utils import read_split, load_graph_data, check_desc, get_edge_adj_matrices

def construct_placeholders(edge_types):
    placeholders = {
        'dropout': tf.placeholder_with_default(0., shape=()),
        # 'current_i': tf.placeholder(tf.int32, shape=(), name='current_i'),
        # 'current_j': tf.placeholder(tf.int32, shape=(), name='current_j'),
        # 'edge_type_idx': tf.placeholder(tf.int32, shape=(), name='edge_type_idx'),
    }
    placeholders.update({'adj_mats_%d,%d' % (i, j): tf.sparse_placeholder(tf.float32) for i, j in edge_types})
    placeholders.update({'norm_%d,%d' % (i, j): tf.placeholder(tf.float32) for i, j in edge_types})
    placeholders.update({'pos_weight_%d,%d' % (i, j): tf.placeholder(tf.float32, shape=()) for i, j in edge_types})
    print(i for i,_ in edge_types)
    placeholders.update({'feat_%d' % i: tf.sparse_placeholder(tf.float32) for i, _ in edge_types})
    return placeholders


def get_roc_score(edges_pos, edges_neg, edge_type, emb=None):
    # print(edge_type)
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        # emb = sess.run(model.z_mean, feed_dict=feed_dict)
        emb = sess.run(model.reconstructions, feed_dict=feed_dict)


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    print(emb[edge_type].shape)

    # Predict on test set of edges
    # for i in emb:
    #     print(emb)
    #     print(emb[i].shape)

    # adj_rec = np.dot(emb, emb.T)
    adj_rec = emb[edge_type]
    preds = []
    pos = []
    # print(adj_rec)
    for e in edges_pos:
        # print(e)

        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_mats_orig[edge_type][e[0], e[1]])
        # pos.append(0)

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_mats_orig[edge_type][e[0], e[1]])
        # neg.append(1)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'model_ae', 'Model string.')
flags.DEFINE_string('optimizer', 'multi', 'Model string.')
flags.DEFINE_string('dataset', 'bio', 'Dataset string.')
flags.DEFINE_string('edge_type', 'drug_gene', 'Type of edges to learn.')
flags.DEFINE_integer('random_seed', 0, 'Random seed')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
optimizer_str = FLAGS.optimizer
dataset_str = FLAGS.dataset
dataset_path = "data/" + dataset_str + "/parsed/"
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

p = dataset_path + "random_splits/" + edge_type + "/random" + str(random_seed) + "/"
G_train, test_positive_e, test_negative_e, val_positive_e, val_negative_e, train_edges = read_split(G, edge_type.split("_"), random_seed, p)

adjs_train = get_edge_adj_matrices(G_train, {et: adjs_orig[et]["nodes"] for et in adjs_orig})

adj_train = nx.to_scipy_sparse_matrix(G_train)
# adj = adj_train

k = [type_num_dict[t] for t in edge_type.split("_")]
print("k", k)
k = tuple(k)
print("k", k)
print(edge_type)
# print(adjs_orig)

# print(adjs_orig[edge_type])
# print(adjs_orig[edge_type]["nodes"])
nodes0 = adjs_orig[edge_type]["nodes"][0]
nodes1 = adjs_orig[edge_type]["nodes"][1]
# print(nodes0)
# print(nodes1)



val_positive = [(nodes0.index(e1), nodes1.index(e2)) for (e1, e2) in val_positive_e]
val_negative = [(nodes0.index(e1), nodes1.index(e2)) for (e1, e2) in val_negative_e]

test_positive = [(nodes0.index(e1), nodes1.index(e2)) for (e1, e2) in test_positive_e]
test_negative = [(nodes0.index(e1), nodes1.index(e2)) for (e1, e2) in test_negative_e]
#
print("Dataset read")

if FLAGS.features == 0:
    # features = sp.identity(G_train.number_of_nodes())  # featureless
    features = {
        0: sparse_to_tuple(sp.identity(519).tocoo()),
        1: sparse_to_tuple(sp.identity(284).tocoo()),
        2: sparse_to_tuple(sp.identity(19820).tocoo()),
    }

    num_feat = { i: features[i][2][1] for i in features}
    nonzero_feat = { i: features[i][1].shape[0] for i in features}

    # num_features = features[2][1]
    # features_nonzero = features[1].shape[0]

print("numfeat", num_feat)
print("nonzero_feat", nonzero_feat)

adj = adj_train


num_nodes = adj_train.shape[0]
#
# features = sparse_to_tuple(features.tocoo())
# num_features = features[2][1]
#
# features_nonzero = features[1].shape[0]
#


# data representation
adj_mats_orig = {
    (1, 2): adjs_orig["drug_gene"]["adj"],
    (2, 1): adjs_orig["drug_gene"]["adj"].transpose(copy=True),
    (0, 2): adjs_orig["disease_gene"]["adj"],
    (2, 0): adjs_orig["disease_gene"]["adj"].transpose(copy=True),
    (2, 2): adjs_orig["gene_gene"]["adj"],
}

adj_mats_train = {
    (1, 2): adjs_train["drug_gene"]["adj"],
    (2, 1): adjs_train["drug_gene"]["adj"].transpose(copy=True),
    (0, 2): adjs_train["disease_gene"]["adj"],
    (2, 0): adjs_train["disease_gene"]["adj"].transpose(copy=True),
    (2, 2): adjs_train["gene_gene"]["adj"],
}

for i,j in adj_mats_train:
    print(i, j, adj_mats_train[i,j].shape)


adj_mats_train_normalized = { et: preprocess_graph(adj_mats_train[et]) for et in adj_mats_train }

edge_type2dim = {(i,j): adj_mats_train[i,j].shape for i,j in adj_mats_train}
print("edgetype2dim", edge_type2dim)

# edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (1, 2): 'bilinear',
    (2, 1): 'bilinear',
    (0, 2): 'bilinear',
    (2, 0): 'bilinear',
    (2, 2): 'bilinear',
}

# edge_type2decoder = {
#     (1, 2): 'innerproduct',
#     (2, 1): 'innerproduct',
#     (0, 2): 'innerproduct',
#     (2, 0): 'innerproduct',
#     (2, 2): 'innerproduct',
# }

edge_types = [et for et in adj_mats_orig]
num_edge_types = len(edge_types)
print(num_edge_types)

edge_type2idx = {}
idx2edge_type = {}
r = 0
for i, j in edge_types:
    edge_type2idx[i, j] = r
    idx2edge_type[r] = i, j
    r += 1

print(edge_type2idx)
print(idx2edge_type)


# idx2edge_type = {}
# edge_type2idx = {}
# r = 0
# for i, j in edge_types:
#     edge_type2idx[i, j] = r
#     idx2edge_type[r] = i, j
#     r += 1

placeholders = construct_placeholders(edge_types)

# Create model
# model = None
# if model_str == 'het_opt':
model = GCNModelAEHet(placeholders,
                      num_feat,
                      nonzero_feat,
                      edge_types=edge_types,
                      decoders=edge_type2decoder)

pos_weights = {et: float(adj_mats_train[et].shape[0] * adj_mats_train[et].shape[0] - adj_mats_train[et].sum()) / adj_mats_train[et].sum() for et in adj_mats_train}
print (pos_weights)

norms = {et: adj_mats_train[et].shape[0] * adj_mats_train[et].shape[0] / float((adj_mats_train[et].shape[0] * adj_mats_train[et].shape[0] - adj_mats_train[et].sum()) * 2) for et in adj_mats_train}
print(norms)

# pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


# Optimizer
with tf.name_scope('optimizer'):
    if optimizer_str == 'single':



        opt = OptimizerAE(
                          preds = model.reconstructions[k],
                          labels = model.adj_mats[k], #{et: placeholders['adj_mats_%d,%d' % (et[0], et[1])] for et in edge_types},
                          # placeholders = placeholders,
        # embeddings=model.embeddings,
        # latent_inters=model.latent_inters,
        # latent_varies=model.latent_varies,
        # degrees=degrees,
        # edge_types=edge_types,
        # edge_type2dim=edge_type2dim,
        # adj_mats = model.adj_mats,
        # reconstructions=model.reconstructions,
        norm = norms[k],
        pos_weight = pos_weights[k],
        # batch_size=FLAGS.batch_size,
        # margin=FLAGS.max_margin
        # edge_type = k


        )
    elif optimizer_str == "multi":
        optimizers = []
        for i in range(len(edge_type2idx)):
            optimizers.append(
                OptimizerAE(
                    preds=model.reconstructions[idx2edge_type[i]],
                    labels=model.adj_mats[idx2edge_type[i]],
                    norm=norms[idx2edge_type[i]],
                    pos_weight=pos_weights[idx2edge_type[i]],
                )
            )

                      # labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                      #                                             validate_indices=False), [-1]),
                      # pos_weight=pos_weight,
                      # norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []
val_roc_score = []

# adj_label = adj_train + sp.eye(adj_train.shape[0])
# adj_label = sparse_to_tuple(adj_label)

print("Trainable variables")
variables_names = [v.name for v in tf.trainable_variables()]
values = sess.run(variables_names)
for k1, v in zip(variables_names, values):
    print("Variable: ", k1)
    print ("Shape: ", v.shape)
    print (v)


# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()

    edge_types_iterator = list(adj_mats_train.keys())

    for idx in idx2edge_type:
        print (idx)


        # Construct feed dictionary
        # feed_dict = construct_feed_dict(adj_norm, adj_label, et, features, placeholders)
        feed_dict = construct_feed_dict(adj_mats_train, adj_mats_train_normalized, features, edge_types, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # print(feed_dict)
        # Run single weight update
        # outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        if optimizer_str == "multi":
            outs = sess.run([optimizers[idx].opt_op, optimizers[idx].cost], feed_dict=feed_dict)
        else:
            outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        # outs = sess.run([], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        # avg_accuracy = outs[2]

        roc_curr, ap_curr = get_roc_score(val_positive, val_negative, k)


        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "val_roc=", "{:.5f}".format(roc_curr),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

        val_roc_score.append(roc_curr)
        # if len(val_roc_score) > 100 and check_desc(val_roc_score[-5:]):  # stop learning if roc dropping
        #     break
        if roc_curr > 0.97:
            break

    # if len(val_roc_score) > 100 and check_desc(val_roc_score[-5:]): # stop learning if roc dropping
    #     break
    if roc_curr > 0.97:
        break

print("Optimization Finished!")

roc_score, ap_score = get_roc_score(test_positive, test_negative, k)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
