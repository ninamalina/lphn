import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


# def _hinge_loss(self, aff, neg_aff):
#     """Maximum-margin optimization using the hinge loss."""
#     diff = tf.nn.relu(tf.subtract(neg_aff, tf.expand_dims(aff, 0) - self.margin), name='diff')
#     loss = tf.reduce_sum(diff)
#     return loss
#
# def _xent_loss(self, aff, neg_aff):
#     """Cross-entropy optimization."""
#
#     true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
#     negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
#     loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
#     return loss


class OptimizerAE(object):
    # def __init__(self, preds, labels, placeholders, edge_types, adj_mats, reconstructions, norms, pos_weights, edge_type ):
    def __init__(self, preds, labels, norm, pos_weight):

        labels = tf.sparse_tensor_to_dense(labels, validate_indices=False)
        # loss = norm * tf.nn.weighted_cross_entropy_with_logits(targets=labels,  logits=preds, pos_weight=pos_weight)
        loss =  tf.nn.sigmoid_cross_entropy_with_logits(targets=labels,  logits=preds)
        # loss =  tf.nn.softmax_cross_entropy_with_logits(targets=labels,  logits=preds)

        self.cost = tf.reduce_mean(loss)

        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.predictions))
        # self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
        #                                    tf.cast(labels_sub, tf.int32))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
