import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class DecagonOptimizer(object):
    def __init__(self, embeddings, latent_inters, latent_varies,
                 edge_types, edge_type2dim, placeholders, degrees=0,
                 margin=0.1, neg_sample_weights=1., batch_size=100, labels=""):
        self.embeddings= embeddings
        self.latent_inters = latent_inters
        self.latent_varies = latent_varies
        self.edge_types = edge_types
        # self.degrees = degrees
        self.edge_type2dim = edge_type2dim
        self.obj_type2n = {i: self.edge_type2dim[i,j][0][0] for i, j in self.edge_types}
        self.margin = margin
        self.neg_sample_weights = neg_sample_weights
        self.batch_size = batch_size
        self.placeholders = placeholders

        self.inputs = placeholders['batch']
        self.edge_type_idx = placeholders['edge_type_idx']
        self.row_edge_type = placeholders['row_edge_type']
        self.col_edge_type = placeholders['col_edge_type']

        self.adj_mat = placeholders['adj_mats_1,2,0']
        print(self.adj_mat)

        self.labels = tf.reshape(tf.sparse_tensor_to_dense(self.adj_mat, validate_indices=False), [-1])

        print(self.labels)
        # self.positive_edges = self.labels.non_zero()
        #
        # self.n = len(self.positive_edges)
        # self.negative_edges = []
        self.row_inputs = tf.squeeze(gather_cols(self.inputs, [0]))
        self.col_inputs = tf.squeeze(gather_cols(self.inputs, [1]))

        obj_type_n = [self.obj_type2n[i] for i in range(len(self.embeddings))]
        self.obj_type_lookup_start = tf.cumsum([0] + obj_type_n[:-1])
        self.obj_type_lookup_end = tf.cumsum(obj_type_n)

        # labels = tf.reshape(tf.cast(self.row_inputs, dtype=tf.int64), [self.batch_size, 1])
        # neg_samples_list = []
        # for i, j in self.edge_types:
        #     for k in range(self.edge_types[i,j]):
        #         neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        #             true_classes=labels,
        #             num_true=1,
        #             num_sampled=self.batch_size,
        #             unique=False,
        #             range_max=len(self.degrees[i][k]),
        #             distortion=0.75,
        #             unigrams=self.degrees[i][k].tolist())
        #         neg_samples_list.append(neg_samples)
        # self.neg_samples = tf.gather(neg_samples_list, self.batch_edge_type_idx)



        # self.preds = self.batch_predict(self.row_inputs, self.col_inputs)
        # self.outputs = tf.diag_part(self.preds)
        # self.outputs = tf.reshape(self.outputs, [-1])
        #
        # self.neg_preds = self.batch_predict(self.neg_samples, self.col_inputs)
        # self.neg_outputs = tf.diag_part(self.neg_preds)
        # self.neg_outputs = tf.reshape(self.neg_outputs, [-1])

        self.predict()  # tuki dobimo self.predictions

        self._build()

    # def batch_predict(self, row_inputs, col_inputs):
    #     concatenated = tf.concat(self.embeddings, 0)
    #
    #     ind_start = tf.gather(self.obj_type_lookup_start, self.batch_row_edge_type)
    #     ind_end = tf.gather(self.obj_type_lookup_end, self.batch_row_edge_type)
    #     indices = tf.range(ind_start, ind_end)
    #     row_embeds = tf.gather(concatenated, indices)
    #     row_embeds = tf.gather(row_embeds, row_inputs)
    #
    #     ind_start = tf.gather(self.obj_type_lookup_start, self.batch_col_edge_type)
    #     ind_end = tf.gather(self.obj_type_lookup_end, self.batch_col_edge_type)
    #     indices = tf.range(ind_start, ind_end)
    #     col_embeds = tf.gather(concatenated, indices)
    #     col_embeds = tf.gather(col_embeds, col_inputs)
    #
    #     latent_inter = tf.gather(self.latent_inters, self.batch_edge_type_idx)
    #     latent_var = tf.gather(self.latent_varies, self.batch_edge_type_idx)
    #
    #     product1 = tf.matmul(row_embeds, latent_var)
    #     product2 = tf.matmul(product1, latent_inter)
    #     product3 = tf.matmul(product2, latent_var)
    #     preds = tf.matmul(product3, tf.transpose(col_embeds))
    #     return preds

    def predict(self):
        concatenated = tf.concat(self.embeddings, 0)

        ind_start = tf.gather(self.obj_type_lookup_start, self.row_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.row_edge_type)
        indices = tf.range(ind_start, ind_end)
        row_embeds = tf.gather(concatenated, indices)

        ind_start = tf.gather(self.obj_type_lookup_start, self.col_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.col_edge_type)
        indices = tf.range(ind_start, ind_end)
        col_embeds = tf.gather(concatenated, indices)

        latent_inter = tf.gather(self.latent_inters, self.edge_type_idx)
        latent_var = tf.gather(self.latent_varies, self.edge_type_idx)

        product1 = tf.matmul(row_embeds, latent_var)
        product2 = tf.matmul(product1, latent_inter)
        product3 = tf.matmul(product2, latent_var)
        self.predictions = tf.matmul(product3, tf.transpose(col_embeds))
        print("predictions")
        print(self.predictions)
        print(self.predictions.shape)

    def _build(self):
        # tuki mormo dodat se en parameter in sicer prave napovedi
        # self.cost = self._hinge_loss(self.outputs, self.neg_outputs)
        # self.cost = self._xent_loss(self.outputs, self.neg_outputs)
        pos_weight = float(self.adj_mat.shape[0] * self.adj_mat.shape[0] - self.adj_mat.sum()) / self.adj_mat.sum()
        norm = self.adj_mat.shape[0] * self.adj_mat.shape[0] / float((self.adj_mat.shape[0] * self.adj_mat.shape[0] - self.adj_mat.sum()) * 2)
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=self.predictions, targets=self.adj_mat, pos_weight=pos_weight))
        # self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=self.predictions, targets=self.adj_mat))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

    def _hinge_loss(self, aff, neg_aff):
        """Maximum-margin optimization using the hinge loss."""
        diff = tf.nn.relu(tf.subtract(neg_aff, tf.expand_dims(aff, 0) - self.margin), name='diff')
        loss = tf.reduce_sum(diff)
        return loss

    def _xent_loss(self, aff, neg_aff):
        """Cross-entropy optimization."""

        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss


def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(
            tf.gather(p_flat, i_flat), [p_shape[0], -1])



class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


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
