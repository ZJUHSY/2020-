import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import getopt
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import json
import numpy as np
import tensorflow as tf






def classify(activations, reuse):
  """Classify block."""
  with tf.variable_scope("Classify", reuse=reuse):
    logits = F.linear(
        activations, units=2, use_bias=False)
    predictions = tf.nn.softmax(logits)
    return logits, predictions

def relational_attention(encoded_queries,
                         candidate_keys,
                         candidate_values, attention_dim=32,
                         normalization="softmax"):
    """Block for dot-product based relational attention."""

    activations = torch.mm(candidate_keys, encoded_queries.transpose(0, 1))
    activations /= np.sqrt(attention_dim)
    activations = activations.transpose(0, 1)
    if normalization == "softmax":
        weight_coefs = F.softmax(activations)
        # print(normalization)
    # elif normalization == "sparsemax":
    #     weight_coefs = torch.FloatTensor(tf.contrib.sparsemax.sparsemax(activations))
        # weight_coefs = torch.FloatTensor(tf.contrib.sparsemax.sparsemax(activations))
    else:
        weight_coefs = activations
    weighted_encoded = torch.mm(weight_coefs, candidate_values)
    return weighted_encoded, weight_coefs

# test
def test(cnn, test_loader, use_cuda):
    pred_v = []
    right, total = 0, 0
    right_neg, total_neg = 0, 0
    right_pos, total_pos = 0, 0
    for step, data in enumerate(test_loader):

        vec, label = data
        if use_cuda:
            vec = vec.cuda()
            label = label.cuda()
        output = cnn(vec)
        pred = torch.max(output, 1)[1]
        pred_v.extend(pred)
        label = label.to(dtype=torch.int64)

        right_neg += label[(pred == label) & (label == 0)].size(0)
        total_neg += label[label == 0].size(0)
        right_pos += label[(pred == label) & (label == 1)].size(0)
        total_pos += label[label == 1].size(0)
        right += label[pred == label].size(0)
        total += label.size(0)
    print('Accuracy:%.3f %d/%d' % (
    float(right_neg + right_pos) / float(total_neg + total_pos), right_neg + right_pos, total_neg + total_pos))
    print('Negative accuracy:%.3f  %d/%d' % (float(right_neg) / float(total_neg), right_neg, total_neg))
    print('Positive accuracy:%.3f  %d/%d' % (float(right_pos) / float(total_pos), right_pos, total_pos))


