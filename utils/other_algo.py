import tensorflow as tf
# import tensorflow_probability as tfp
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras import backend as K
import math
import statistics
import numpy as np
def cosine_similarity(tensor1, tensor2):
    # Normalize the tensors
    tensor1_normalized = tf.nn.l2_normalize(tensor1, axis=-1)
    tensor2_normalized = tf.nn.l2_normalize(tensor2, axis=-1)

    # Calculate the cosine similarity
    cos_similarity = K.mean(CosineSimilarity(axis=-1)(tensor1_normalized, tensor2_normalized))

    return cos_similarity


def model_cosine_similarity(weight1, weight2):
    bla=[]
    for i in range(len(weight1)):
        bla.append(cosine_similarity(weight1[i], weight2[i]))
    return sum(bla) / len(bla)


def krum(weight_list):
    bla = []

    for i in range(len(weight_list)):
        sla = []
        for j in range(len(weight_list)):
            sla.append(model_cosine_similarity(weight_list[i], weight_list[j]))
        bla.append(sum(sla) / len(sla))
    print(min(bla))
    return weight_list[bla.index(min(bla))]


def trimmed_mean(weight, trim_percent):
    num_trim = math.ceil(len(weight) * trim_percent)
    final_weight = []
    for i in range(len(weight[0])):
        tensor_list = []
        for j in range(len(weight)):
            tensor_list.append(weight[j][i])
        # Convert the list of tensors to a TensorFlow tensor
        stacked_tensors = tf.stack(tensor_list)

        # Sort the elements along the specified axis
        sorted_tensors = tf.sort(stacked_tensors, axis=0)

        # Remove elements from both ends
        trimmed_values = sorted_tensors[num_trim:-num_trim, :]

        # Calculate the mean of the trimmed values
        trimmed_mean_tensor = tf.reduce_mean(trimmed_values, axis=0)

        final_weight.append(trimmed_mean_tensor)

    return final_weight


# def median(weight):
#     final_weight = []
#     for i in range(len(weight[0])):
#         tensor_list = []
#         for j in range(len(weight)):
#             tensor_list.append(weight[j][i])
#         # Convert the list of tensors to a TensorFlow tensor
#         stacked_tensors = tf.stack(tensor_list)
#
#         median_tensor = tfp.stats.percentile(stacked_tensors, q=50.0, axis=0)
#
#         final_weight.append(median_tensor)
#
#     return final_weight

def median(weight):
    final_weight = []
    for i in range(len(weight[0])):
        tensor_list = []
        for j in range(len(weight)):
            tensor_list.append(weight[j][i])

        # Convert the list of tensors to a TensorFlow tensor
        stacked_tensors = tf.stack(tensor_list)

        # Sort the elements along the specified axis
        sorted_tensors = tf.sort(stacked_tensors, axis=0)

        # Calculate the median index
        median_index = sorted_tensors.shape[0] // 2

        # Get the median value(s)
        if sorted_tensors.shape[0] % 2 == 0:
            # If even number of elements, take the average of the two middle elements
            median_value = (sorted_tensors[median_index - 1] + sorted_tensors[median_index]) / 2.0
        else:
            # If odd number of elements, simply take the middle element
            median_value = sorted_tensors[median_index]

        final_weight.append(median_value)

    return final_weight

def euclidean_distance(tensor1, tensor2):
    # Normalize the tensors
    tensor1_normalized = tf.nn.l2_normalize(tensor1, axis=-1)
    tensor2_normalized = tf.nn.l2_normalize(tensor2, axis=-1)

    # Calculate the cosine similarity
    euclidean_distance = tf.norm(tensor1_normalized  - tensor2_normalized)

    return euclidean_distance

def model_euclidean_distance(weight1, weight2):
    bla= 0
    for i in range(len(weight1)):
        bla+= euclidean_distance(weight1[i], weight2[i])
    return bla


def krum_new(weight_list, percent):
    num_trim = math.ceil(len(weight_list) * percent)
    bla = []

    for i in range(len(weight_list)):
        sla = []
        for j in range(len(weight_list)):
            sla.append(model_euclidean_distance(weight_list[i], weight_list[j]))
        sorted_list = sorted(sla)

        # Get the lowest 10 values
        sla = sorted_list[:-num_trim]
        bla.append(sum(sla))
    print(min(bla))
    return weight_list[bla.index(min(bla))]

def fedasl(data, alpha, beta):
    val= np.zeros(len(data))
    for i in range(len(data)):
        if abs(data[i] - statistics.median(data))<=statistics.stdev(data)*alpha:
            val[i] = statistics.stdev(data)*beta
        else:
            val[i]= abs(data[i] - statistics.median(data))
    weighted_value= np.zeros(len(data))
    for j in range(len(val)):
        weighted_value[j]= np.reciprocal(val[j])/np.sum(np.reciprocal(val))
    return weighted_value, statistics.median(data),statistics.stdev(data)

