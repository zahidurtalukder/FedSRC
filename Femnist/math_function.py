import tensorflow as tf
import numpy as np
import statistics
import math

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    # get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clinets
    global_count = sum \
        ([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names] ) *bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy( ) *bs
    return local_count /global_count

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad

def fed_avg_weight(a):
    b =list()
    for i in range(len(a)):
        b.append(a[i]/sum(a))
    return b
def weight_std_dev_median(data):
    val= np.zeros(len(data))
    for i in range(len(data)):
        if abs(data[i] - statistics.median(data))<=statistics.stdev(data)*1:
            val[i] = statistics.stdev(data)*.5
#         elif statistics.stdev(data)*1 <= abs(data[i] - statistics.median(data))<=statistics.stdev(data)*2:
#             val[i] = statistics.stdev(data)*1.5
        else:
            val[i]= abs(data[i] - statistics.median(data))
    weighted_value= np.zeros(len(data))
    for j in range(len(val)):
        weighted_value[j]= np.reciprocal(val[j])/np.sum(np.reciprocal(val))
    return weighted_value

def median_std(data):
    return round(statistics.median(data)*100,4),round(statistics.stdev(data)*100,4), round(statistics.mean(data)*100,4)

def weight_alpha_beta_median(data, alpha, beta):
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

def weight_alpha_beta_median_loss(data, alpha, beta):
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


def median_algo_mnist(client_weights):
    weight = np.zeros(client_weights[0][0].shape)
    bias = np.zeros(client_weights[0][1].shape)
    a, b = client_weights[0][0].shape
    for i in range(a):
        for j in range(b):
            int_list = []
            for k in range(len(client_weights)):
                int_list.append(client_weights[k][0][i][j])

            weight[i][j] = np.median(int_list)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][1][m])

        bias[m] = np.median(int_list)
    new = [weight, bias]
    return new


def trimmed_mean_algo_mnist(client_weights, percent_trim):
    num_trim = math.ceil(len(client_weights) * percent_trim)
    one_side = math.ceil(num_trim / 2)
    weight = np.zeros(client_weights[0][0].shape)
    bias = np.zeros(client_weights[0][1].shape)
    a, b = client_weights[0][0].shape
    for i in range(a):
        for j in range(b):
            int_list = []
            for k in range(len(client_weights)):
                int_list.append(client_weights[k][0][i][j])
            int_list = np.sort(int_list)
            weight[i][j] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][1][m])

        int_list = np.sort(int_list)
        bias[m] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)
    new = [weight, bias]
    return new


def median_algo_cifar10(client_weights):
    # 1st layer
    a, b, c, d = client_weights[0][0].shape
    weight_1 = np.zeros([a, b, c, d])
    bias_1 = np.zeros([d, ])

    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    int_list = []
                    for m in range(len(client_weights)):
                        int_list.append(client_weights[m][0][i][j][k][l])
                    weight_1[i][j][k][l] = np.median(int_list)

    for m in range(d):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][1][m])
        bias_1[m] = np.median(int_list)

    # 2nd layer
    a, b, c, d = client_weights[0][2].shape
    weight_2 = np.zeros([a, b, c, d])
    bias_2 = np.zeros([d, ])

    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    int_list = []
                    for m in range(len(client_weights)):
                        int_list.append(client_weights[m][2][i][j][k][l])
                    weight_2[i][j][k][l] = np.median(int_list)

    for m in range(d):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][3][m])
        bias_2[m] = np.median(int_list)

    # 3rd layer
    a, b = client_weights[0][4].shape
    weight_3 = np.zeros([a, b])
    bias_3 = np.zeros([b, ])

    for i in range(a):
        for j in range(b):
            int_list = []
            for m in range(len(client_weights)):
                int_list.append(client_weights[m][4][i][j])
            weight_3[i][j] = np.median(int_list)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][5][m])
        bias_3[m] = np.median(int_list)

    # 4th layer
    a, b = client_weights[0][6].shape
    weight_4 = np.zeros([a, b])
    bias_4 = np.zeros([b, ])

    for i in range(a):
        for j in range(b):
            int_list = []
            for m in range(len(client_weights)):
                int_list.append(client_weights[m][6][i][j])
            weight_4[i][j] = np.median(int_list)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][7][m])
        bias_4[m] = np.median(int_list)

    new = [weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, weight_4, bias_4]
    return new


def trimmed_mean_algo_cifar10(client_weights, percent_trim):
    num_trim = math.ceil(len(client_weights) * percent_trim)
    one_side = math.ceil(num_trim / 2)
    # 1st layer
    a, b, c, d = client_weights[0][0].shape
    weight_1 = np.zeros([a, b, c, d])
    bias_1 = np.zeros([d, ])

    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    int_list = []
                    for m in range(len(client_weights)):
                        int_list.append(client_weights[m][0][i][j][k][l])
                    int_list = np.sort(int_list)
                    weight_1[i][j][k][l] = sum(int_list[one_side: (len(int_list) - one_side)]) / (
                                len(int_list) - 2 * one_side)

    for m in range(d):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][1][m])
        int_list = np.sort(int_list)
        bias_1[m] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    # 2nd layer
    a, b, c, d = client_weights[0][2].shape
    weight_2 = np.zeros([a, b, c, d])
    bias_2 = np.zeros([d, ])

    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    int_list = []
                    for m in range(len(client_weights)):
                        int_list.append(client_weights[m][2][i][j][k][l])
                    int_list = np.sort(int_list)
                    weight_2[i][j][k][l] = sum(int_list[one_side: (len(int_list) - one_side)]) / (
                                len(int_list) - 2 * one_side)

    for m in range(d):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][3][m])
        int_list = np.sort(int_list)
        bias_2[m] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    # 3rd layer
    a, b = client_weights[0][4].shape
    weight_3 = np.zeros([a, b])
    bias_3 = np.zeros([b, ])

    for i in range(a):
        for j in range(b):
            int_list = []
            for m in range(len(client_weights)):
                int_list.append(client_weights[m][4][i][j])
            int_list = np.sort(int_list)
            weight_3[i][j] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][5][m])
        int_list = np.sort(int_list)
        bias_3[m] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    # 4th layer
    a, b = client_weights[0][6].shape
    weight_4 = np.zeros([a, b])
    bias_4 = np.zeros([b, ])

    for i in range(a):
        for j in range(b):
            int_list = []
            for m in range(len(client_weights)):
                int_list.append(client_weights[m][6][i][j])
            int_list = np.sort(int_list)
            weight_4[i][j] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][7][m])
        int_list = np.sort(int_list)
        bias_4[m] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    new = [weight_1, bias_1, weight_2, bias_2, weight_3, bias_3, weight_4, bias_4]
    return new


def median_algo_femnist(client_weights):
    # 1st layer
    a, b = client_weights[0][0].shape
    weight_1 = np.zeros([a, b])
    bias_1 = np.zeros([b, ])

    for i in range(a):
        for j in range(b):
            int_list = []
            for m in range(len(client_weights)):
                int_list.append(client_weights[m][0][i][j])
            weight_1[i][j] = np.median(int_list)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][1][m])
        bias_1[m] = np.median(int_list)

    # 2nd layer
    a, b = client_weights[0][2].shape
    weight_2 = np.zeros([a, b])
    bias_2 = np.zeros([b, ])

    for i in range(a):
        for j in range(b):
            int_list = []
            for m in range(len(client_weights)):
                int_list.append(client_weights[m][2][i][j])
            weight_2[i][j] = np.median(int_list)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][3][m])
        bias_2[m] = np.median(int_list)

    new = [weight_1, bias_1, weight_2, bias_2]
    return new


def trimmed_mean_algo_femnist(client_weights, percent_trim):
    num_trim = math.ceil(len(client_weights) * percent_trim)
    one_side = math.ceil(num_trim / 2)

    # 1st layer
    a, b = client_weights[0][0].shape
    weight_1 = np.zeros([a, b])
    bias_1 = np.zeros([b, ])

    for i in range(a):
        for j in range(b):
            int_list = []
            for m in range(len(client_weights)):
                int_list.append(client_weights[m][0][i][j])
            int_list = np.sort(int_list)
            weight_1[i][j] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][1][m])
        int_list = np.sort(int_list)
        bias_1[m] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    # 4th layer
    a, b = client_weights[0][2].shape
    weight_2 = np.zeros([a, b])
    bias_2 = np.zeros([b, ])

    for i in range(a):
        for j in range(b):
            int_list = []
            for m in range(len(client_weights)):
                int_list.append(client_weights[m][2][i][j])
            int_list = np.sort(int_list)
            weight_2[i][j] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    for m in range(b):
        int_list = []
        for k in range(len(client_weights)):
            int_list.append(client_weights[k][3][m])
        int_list = np.sort(int_list)
        bias_2[m] = sum(int_list[one_side: (len(int_list) - one_side)]) / (len(int_list) - 2 * one_side)

    new = [weight_1, bias_1, weight_2, bias_2]
    return new