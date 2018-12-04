from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class MultiLayer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, edge_type=(), num_types=-1, **kwargs):
        self.edge_type = edge_type
        self.num_types = num_types
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs



class GraphConvolutionMulti(MultiLayer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj_mats, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionMulti, self).__init__(**kwargs)
        self.adj_mats = adj_mats
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')

    def _call(self, inputs):

        # x = tf.nn.dropout(inputs, 1-self.dropout)
        # x = tf.matmul(x, self.vars['weights'])
        # x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type], x)
        # outputs.append(self.act(x))
        # outputs = tf.add_n(outputs)
        # outputs = tf.nn.l2_normalize(outputs, dim=1)

        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type], x)
        outputs = self.act(x)

        return outputs

class GraphConvolutionSparseMulti(MultiLayer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj_mats,
                 nonzero_feat, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseMulti, self).__init__(**kwargs)
        self.dropout = dropout
        self.adj_mats = adj_mats
        self.act = act
        self.issparse = True
        self.nonzero_feat = nonzero_feat
        print(self.edge_type)
        with tf.variable_scope('%s_vars' % self.name):
            # for k in range(self.num_types):
            self.vars['weights'] = weight_variable_glorot(
                    input_dim[self.edge_type[1]], output_dim, name='weights')

    def _call(self, inputs):
        # print(self.adj_mats)
        # print(self.num_types)
        # outputs = []
        # for k in range(self.num_types):
        #     x = dropout_sparse(inputs, 1-self.dropout, self.nonzero_feat[self.edge_type[1]])
        #     x = tf.sparse_tensor_dense_matmul(x, self.vars['weights_%d' % k])
        #     print(self.adj_mats[self.edge_type][k])
        #     x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
        #     outputs.append(self.act(x))
        # outputs = tf.add_n(outputs)
        # outputs = tf.nn.l2_normalize(outputs, dim=1)
        #
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.nonzero_feat[self.edge_type[1]])
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type], x)
        outputs = self.act(x)
        return outputs


class DistMultDecoder(MultiLayer):
    """DistMult Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DistMultDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            # for k in range(self.num_types):
            tmp = weight_variable_glorot(input_dim, 1, name='relation')
            self.vars['relation'] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type

        inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
        inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
        relation = tf.diag(self.vars['relation'])
        intermediate_product = tf.matmul(inputs_row, relation)
        rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
        # outputs.append(self.act(rec))
        return self.act(rec)


class BilinearDecoder(MultiLayer):
    """Bilinear Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(BilinearDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['relation'] = weight_variable_glorot(input_dim, input_dim, name='relation')

    def _call(self, inputs):
        i, j = self.edge_type
        inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
        inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
        intermediate_product = tf.matmul(inputs_row, self.vars['relation'])
        rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
        outputs = self.act(rec)
        return outputs



class InnerProductDecoder(MultiLayer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        i, j = self.edge_type
        inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
        inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
        rec = tf.matmul(inputs_row, tf.transpose(inputs_col))
        outputs = self.act(rec)
        return outputs