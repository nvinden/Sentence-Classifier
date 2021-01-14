import torch
import torch.nn as nn
import torch.functional as F
from gensim import models

w = models.KeyedVectors.load_word2vec_format('/home/nvinden/ML/CNNFORNLP2014/google_embeddings/GoogleNews-vectors-negative300.bin', binary=True)
google_vector_size = w.vector_size
google_data_length = int(w.vectors.size/google_vector_size)

idx_2_embeddings = torch.Tensor(w.vectors)
idx_2_word = w.index2word
idx_2_entity = w.index2entity

class network(nn.Module):
    def __init__(self, sentence_length, embedding_dimension = 5, trained_words = True, static_words = False,
    filter_windows=(5,4,3), feature_maps=100, dropout_rate = 0.5, l2_constraint = 3, mini_batch_size=50):
        super(network, self).__init__()
        
        self.feature_maps = feature_maps
        self.dropout_rate = dropout_rate
        self.l2_constraint = l2_constraint
        self.mini_batch_size = mini_batch_size

        self.trained_words = trained_words
        self.static_words = static_words

        self.convolution = []
        #Defining the convolutional layer of the algorithm
        if trained_words and static_words: #Two layer architecture where one is learned and one is not
            self.embedding_t = nn.Embedding.from_pretrained(idx_2_embeddings)
            self.embedding_s = nn.Embedding.from_pretrained(idx_2_embeddings)
            self.embedding_s.requires_grad_ = False
            for i in range(len(filter_windows)):
                self.convolution.append([nn.Conv3d(2, sentence_length - filter_windows[i] + 1, 1) for j in range(feature_maps)])
        elif trained_words: #Trained layer selected
            self.embedding_t = nn.Embedding.from_pretrained(idx_2_embeddings)
            for i in range(len(filter_windows)):
                self.convolution.append([nn.Conv3d(1, sentence_length - filter_windows[i] + 1, 1) for j in range(feature_maps)])
        elif static_words: #Static layer selected
            self.embedding_s = nn.Embedding.from_pretrained(idx_2_embeddings)
            self.embedding_s.requires_grad_ = False
            for i in range(len(filter_windows)):
                self.convolution.append([nn.Conv3d(1, sentence_length - filter_windows[i] + 1, 1) for j in range(feature_maps)])
        else:
            raise Exception("There must be at least one convolutional layer defined")
        
        print("returned")

    def forward(self, input):
        return self.linear(input)

net = network(20, static_words=True, trained_words=True, feature_maps=5)
print(net)