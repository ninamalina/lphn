from layers import GraphConvolutionMulti, GraphConvolutionSparseMulti, InnerProductDecoder, DistMultDecoder, BilinearDecoder
import tensorflow as tf
from collections import defaultdict

flags = tf.app.flags
FLAGS = flags.FLAGS




class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAEHet(Model):

    def __init__(self, placeholders, num_features, features_nonzero, edge_types, decoders, **kwargs):
        super(GCNModelAEHet, self).__init__(**kwargs)
        self.edge_types = edge_types
        print (edge_types)
        self.num_edge_types = sum(self.edge_types.values())
        self.num_obj_types = max([i for i, _ in self.edge_types]) + 1
        self.decoders = decoders
        self.inputs = {i: placeholders['feat_%d' % i] for i, _ in self.edge_types}
        self.input_dim = num_features
        self.nonzero_feat = features_nonzero
        self.placeholders = placeholders
        self.dropout = placeholders['dropout']
        self.adj_mats = {et: [
            placeholders['adj_mats_%d,%d,%d' % (et[0], et[1], k)] for k in range(n)]
                         for et, n in self.edge_types.items()}
        self.build()

    def _build(self):
        self.hidden1 = defaultdict(list)
        for i, j in self.edge_types:
            print(i, j)

            self.hidden1[i].append(GraphConvolutionSparseMulti(
                input_dim=self.input_dim, output_dim=FLAGS.hidden1,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, nonzero_feat=self.nonzero_feat,
                act=lambda x: x, dropout=self.dropout,
                logging=self.logging)(self.inputs[j]))

        for i, hid1 in self.hidden1.items():
            self.hidden1[i] = tf.nn.relu(tf.add_n(hid1))


        self.embeddings_reltyp = defaultdict(list)
        for i, j in self.edge_types:
            self.embeddings_reltyp[i].append(GraphConvolutionMulti(
                input_dim=FLAGS.hidden1, output_dim=FLAGS.hidden2,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.hidden1[j]))

        self.embeddings = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp.items():
            # self.embeddings[i] = tf.nn.relu(tf.add_n(embeds))
            self.embeddings[i] = tf.add_n(embeds)

        print(self.embeddings_reltyp.items())
        print(len(self.embeddings))

        self.z_mean = self.embeddings

        self.edge_type2decoder = {}
        for i, j in self.edge_types:
            decoder = self.decoders[i, j]
            if decoder == 'innerproduct':
                self.edge_type2decoder[i, j] = InnerProductDecoder(
                    input_dim=FLAGS.hidden2, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'distmult':
                self.edge_type2decoder[i, j] = DistMultDecoder(
                    input_dim=FLAGS.hidden2, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'bilinear':
                self.edge_type2decoder[i, j] = BilinearDecoder(
                    input_dim=FLAGS.hidden2, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)

            # self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
            #                               act=lambda x: x,
            #                               logging=self.logging)(self.embeddings)

        self.reconstructions = []

        self.latent_inters = []
        self.latent_varies = []
        for edge_type in self.edge_types:
            decoder = self.decoders[edge_type]
            for k in range(self.edge_types[edge_type]):
                if decoder == 'innerproduct':
                    glb = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
                    loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
                elif decoder == 'distmult':
                    glb = tf.diag(self.edge_type2decoder[edge_type].vars['relation_%d' % k])
                    loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
                elif decoder == 'bilinear':
                    glb = self.edge_type2decoder[edge_type].vars['relation_%d' % k]
                    loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
                elif decoder == 'dedicom':
                    glb = self.edge_type2decoder[edge_type].vars['global_interaction']
                    loc = tf.diag(self.edge_type2decoder[edge_type].vars['local_variation_%d' % k])
                else:
                    raise ValueError('Unknown decoder type')

                self.latent_inters.append(glb)
                self.latent_varies.append(loc)


# class GCNModelVAE(Model):
#     def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
#         super(GCNModelVAE, self).__init__(**kwargs)
#
#         self.inputs = placeholders['features']
#         self.input_dim = num_features
#         self.features_nonzero = features_nonzero
#         self.n_samples = num_nodes
#         self.adj = placeholders['adj']
#         self.dropout = placeholders['dropout']
#         self.build()
#
#     def _build(self):
#         self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
#                                               output_dim=FLAGS.hidden1,
#                                               adj=self.adj,
#                                               features_nonzero=self.features_nonzero,
#                                               act=tf.nn.relu,
#                                               dropout=self.dropout,
#                                               logging=self.logging)(self.inputs)
#
#         self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
#                                        output_dim=FLAGS.hidden2,
#                                        adj=self.adj,
#                                        act=lambda x: x,
#                                        dropout=self.dropout,
#                                        logging=self.logging)(self.hidden1)
#
#         self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
#                                           output_dim=FLAGS.hidden2,
#                                           adj=self.adj,
#                                           act=lambda x: x,
#                                           dropout=self.dropout,
#                                           logging=self.logging)(self.hidden1)
#
#         self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)
#
#         self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
#                                       act=lambda x: x,
#                                       logging=self.logging)(self.z)
