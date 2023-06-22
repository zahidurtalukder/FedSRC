from Code.utils.client_creation import create_client, creating_flipping_clients, creating_shuffling_clients,creating_noisy_clients_cifar10, creating_targetted_clients, creating_targetted_shuffling_clients, creating_targetted_flipping_clients
import tensorflow as tf
import numpy as np
import keras
import pickle


def cifar10_shuffle_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')


    # Creating bad clients
    clients, bad_client = creating_shuffling_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    file_name = f"../../Data/cifar10/Dataset{client_percent}_{data_percent}_{num_clients}_shuffle_cifar10.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

def cifar10_flip_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')

    # sample_list=[dataset, client_names]
    # file_name = f"../Data/cifar10/Dataset_{num_clients}_cifar10.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()

    # Creating bad clients
    clients, bad_client = creating_flipping_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    file_name = f"../../Data/cifar10/Dataset{client_percent}_{data_percent}_{num_clients}_flip_cifar10.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

def cifar10_noise_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')

    # sample_list=[dataset, client_names]
    # file_name = f"../Data/cifar10/Dataset_{num_clients}_cifar10.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()

    # Creating bad clients
    clients, bad_client = creating_noisy_clients_cifar10(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    file_name = f"../../Data/cifar10/Dataset{client_percent}_{data_percent}_{num_clients}_noise_cifar10.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

    def cifar10_shuffle_data(client_percent, data_percent, num_clients):  # Get the data.
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        nb_classes = 10
        y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
        dataset, client_names = create_client(x_train, y_train, num_clients, initial='client')

        # Creating bad clients
        clients, bad_client = creating_shuffling_clients(dataset, client_names, client_percent, data_percent)

        sample_list = [clients, bad_client, x_test, y_test]
        file_name = f"../../Data/cifar10/Dataset{client_percent}_{data_percent}_{num_clients}_shuffle_cifar10.pkl"

        open_file = open(file_name, "wb")
        pickle.dump(sample_list, open_file)
        open_file.close()

def cifar10_target_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')


    # Creating bad clients
    clients, bad_client = creating_targetted_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    file_name = f"../../Data/cifar10/Dataset{client_percent}_{data_percent}_{num_clients}_target_cifar10.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

def cifar10_target_shuffling_data(client_percent, data_percent, num_clients, percent):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')


    # Creating bad clients
    clients, bad_client = creating_targetted_shuffling_clients(dataset,client_names,client_percent, data_percent,percent)

    sample_list=[clients, bad_client, x_test, y_test]
    file_name = f"../../Data/cifar10/Dataset{client_percent}_{data_percent}_{num_clients}_{percent}_target_shuffling_cifar10.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()

def cifar10_target_flipping_data(client_percent, data_percent, num_clients, percent):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')


    # Creating bad clients
    clients, bad_client = creating_targetted_flipping_clients(dataset,client_names,client_percent, data_percent,percent)

    sample_list=[clients, bad_client, x_test, y_test]
    file_name = f"../../Data/cifar10/Dataset{client_percent}_{data_percent}_{num_clients}_{percent}_target_flipping_cifar10.pkl"

    open_file = open(file_name, "wb")
    pickle.dump(sample_list, open_file)
    open_file.close()