from Code.utils.client_creation import create_client, creating_flipping_clients, creating_shuffling_clients, creating_noisy_clients_mnist
import tensorflow as tf
import numpy as np
import keras
import pickle


def mnist_shuffle_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')


    # Creating bad clients
    clients, bad_client = creating_shuffling_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_shuffle_mnist.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

def mnist_flip_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')


    # Creating bad clients
    clients, bad_client = creating_flipping_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_flip_mnist.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

def mnist_noise_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')

    # Creating bad clients
    clients, bad_client = creating_noisy_clients_mnist(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_noise_mnist.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()