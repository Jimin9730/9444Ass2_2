#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    processed = sample.split()
    
    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    ratingOutput = (ratingOutput > 0.5).long()
    return ratingOutput, categoryOutput.argmax(dim=1)

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.rnn_rate = tnn.LSTM(input_size = 300, hidden_size = 50, num_layers = 2, batch_first = True, bidirectional = True, dropout = 0.2)
        # 50 * 2
        self.fc1_rate = tnn.Linear(50 * 2, 50)
        # output_size = 1
        self.fc2_rate = tnn.Linear(50, 1)
        self.rnn_category = tnn.LSTM(input_size = 300, hidden_size = 50, num_layers = 2, batch_first = True, bidirectional = True, dropout = 0.2)
        self.fc1_category = tnn.Linear(50 * 2, 50)
        self.fc2_category = tnn.Linear(50, 5)
        #self.relu = tnn.ReLU()
        self.dropout = tnn.Dropout(0.2)
        self.sigmoid = tnn.Sigmoid()
        # self.softmax = tnn.Softmax(dim = 1)

    def forward(self, input, length):
        #packed_embedded = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first = True)
        output_rate, (hidden_rate, cell_rate) = self.rnn_rate(input)
        #last_hidden_rate = torch.cat((hidden_rate[-2,:,:], hidden_rate[-1,:,:]), dim = 1)
        last_hidden_rate = self.dropout(torch.cat((hidden_rate[-2,:,:], hidden_rate[-1,:,:]), dim = 1))
        feature_rate = self.fc1_rate(last_hidden_rate)
        #feature_rate = self.dropout(feature_rate)
        feature_rate = self.dropout(self.fc2_rate(feature_rate))
        output_rate = self.sigmoid(feature_rate)
        #output_rate = self.sigmoid(self.fc2_rate(feature_rate))
        output_rate = output_rate.squeeze()
        
        output_category, (hidden_category, cell_category) = self.rnn_category(input)
        # 这里也不用dropout
        last_hidden_category = torch.cat((hidden_category[-2,:,:], hidden_category[-1,:,:]), dim = 1)
        #last_hidden_category = self.dropout(torch.cat((hidden_category[-2,:,:], hidden_category[-1,:,:]), dim = 1))
        #这里换成sigmoid
        feature_category = self.fc1_category(last_hidden_category)
        #feature_category = self.dropout(feature_category)
        #output_category = self.dropout(feature_category)
        output_category = self.dropout(self.fc2_category(feature_category))
        #output_category = self.fc2_category(feature_category)
        
        return output_rate, output_category

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.rate_loss = tnn.BCELoss()
        self.category_loss = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingTarget = ratingTarget.float()
        return self.rate_loss(ratingOutput, ratingTarget) + self.category_loss(categoryOutput, categoryTarget)

net = network()
# net = network(vocab_size, embedding_size, output_size, pad_idx, hidden_size, dropout)
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 7
# optimiser = toptim.SGD(net.parameters(), lr=0.01)
# lr = 0.1
optimiser = toptim.Adam(net.parameters(), lr=0.01)
