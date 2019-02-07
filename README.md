# Link prediction in heterogenous networks

Project contains implementation four different methods used for link prediction in heterogenous networks:

- simple_model.py
Construct simple features of node pairs and use logistic regression to classify the pairs into "edge" or "no edge"

- meta_model.py
Use metapath2vec to embed nodes into vector space, use logistic regression on computed features.

- gae_model.py
Graph autoencoder model on homogenous network- jointly optimize encoder and decoder function, such that embeddings of nodes can be used for link prediction

- gae_het_model.py
Graph autoencoder model modified to work on heterogenous networks
