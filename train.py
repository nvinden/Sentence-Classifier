import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from gensim import models
import numpy as np
from torchtext import data
from torchtext import datasets
import random

w = models.KeyedVectors.load_word2vec_format('/home/nvinden/ML/CNNFORNLP2014/google_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
google_vector_size = w.vector_size
google_data_length = int(w.vectors.size/google_vector_size)


idx_2_embeddings = torch.Tensor(w.vectors)
idx_2_word = w.index2word
idx_2_entity = w.index2entity

#hyper parameters
hp_batch_size = 50
hp_filters = (5,4,3)
hp_feature_maps = 100
hp_dropout_rate = 0.5
hp_l2_constraint = 3
hp_mini_batch_size = 50
hp_learning_rate = 0.05

#model parameters
mp_max_sentence_length = 500
mp_n_epochs = 100
mp_embedding_dim = 300
mp_num_tags = 2

# set up fields
TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=mp_max_sentence_length)
LABEL = data.Field(sequential=False)

# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)

# build the vocabulary
TEXT.build_vocab(train)
LABEL.build_vocab(train)

# make iterator for splits
train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=hp_batch_size, device=0)
train_iter.train = False

tag_list = ("positive", "negative")


def make_tags(tags):
    tag_2_idx = {tag: (i) for i, tag in enumerate(tags)}
    idx_2_tag = {(i): tag for i, tag in enumerate(tags)}
    return tag_2_idx, idx_2_tag

def prepare_labels(labels):
    out = labels - 1
    return out


class network(nn.Module):
    def __init__(self, sentence_length, embedding_dimension = 300, trained_words = True, static_words = False,
    filter_windows=(5,4,3), feature_maps=100, dropout_rate = 0.5, l2_constraint = 3, mini_batch_size=50, num_tags = 2):
        super(network, self).__init__()
        #Hyper Parameters of Model
        self.feature_maps = feature_maps
        self.dropout_rate = dropout_rate
        self.l2_constraint = l2_constraint
        self.mini_batch_size = mini_batch_size
        self.trained_words = trained_words
        self.static_words = static_words
        self.sentence_length = sentence_length
        self.embedding_dimension = embedding_dimension
        self.filter_windows = filter_windows
        self.num_tags = num_tags

        #Network Layers
        self.embedding = nn.Embedding.from_pretrained(idx_2_embeddings)
        self.convolution = []
        for i in range(len(filter_windows)):
            self.convolution.append(nn.Conv2d(1, feature_maps, (filter_windows[i], embedding_dimension)))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear = nn.Linear(feature_maps * len(filter_windows), num_tags)

    def forward(self, input, test=False):
        mini_batch_size = len(input) #number of rows -> mini batch size inputted
        embedding = self.embedding(input)
        linear_inp = torch.empty(mini_batch_size, 0)
        for i in range(len(self.filter_windows)):
            convolution = F.relu(self.convolution[i](embedding.view(mini_batch_size, 1, self.sentence_length, self.embedding_dimension)))
            maxed_out = torch.max(convolution, dim = 2)[0].view(mini_batch_size, self.feature_maps)
            linear_inp = torch.cat((linear_inp, maxed_out), dim=1)
        if not test: #training
            dropout = self.dropout(linear_inp)
            out = F.log_softmax(self.linear(dropout), dim=1)
        else: #testing
            scaled_linear = self.linear
            scaled_linear.weight = scaled_linear.weight * self.dropout_rate

        return out

    def rescale_linear(self, norm_length, weight_norm):
        self.linear.weight = norm_length * (self.linear.weight / weight_norm)

    #input is torch tensors of size number of sentences * batch
    #tag is a vector of shape 1 x batch size
    def cost(self, input, tag):
        y_bar = self.forward(input)
        y = torch.zeros(self.num_tags, 1)
        y[tag_2_idx[tag]] = 1

tag_2_idx, idx_2_tag = make_tags(tag_list)
net = network(mp_max_sentence_length, embedding_dimension=mp_embedding_dim, filter_windows=hp_filters, feature_maps=hp_feature_maps, dropout_rate=hp_dropout_rate, l2_constraint=hp_l2_constraint, mini_batch_size=hp_mini_batch_size, num_tags=mp_num_tags)

net.cost(torch.LongTensor(hp_batch_size, mp_max_sentence_length).random_(0, google_data_length), "positive")

loss = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=hp_learning_rate)

for epoch in range(mp_n_epochs):
    for i, batch in enumerate(train_iter):
        net.zero_grad()
        out = net(batch.text[0])

        prepared_labels = prepare_labels(batch.label)

        loss_value = loss(out, prepared_labels)
        loss_value.backward()
        optimizer.step()

        weight_norm = torch.norm(net.linear.weight)
        if weight_norm > net.l2_constraint:
            print("Pre norm: {}".format(weight_norm))
            net.rescale_linear(net.l2_constraint, weight_norm)
            print("Post norm: {}".format(weight_norm))

        if i % 10 == 0:
            print("Epoch: {} Batch Number: {} Loss: {} Weight norm: {}".format(epoch + 1, i, loss_value, weight_norm))
    torch.save(net.state_dict(), "./CNNFORNLP2014/model")
