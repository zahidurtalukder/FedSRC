from utils.client_creation import create_client, creating_flipping_clients, creating_shuffling_clients, creating_noisy_clients_mnist
import tensorflow as tf
import numpy as np
import keras
import pickle


def mnist_shuffle_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
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
    # file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_shuffle_mnist.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()
    return sample_list


def create_non_iid_mnist(x_train, y_train, num_client, percent,initial):
    # Define the number of shards and clients

    NUM_SHARDS = num_client
    NUM_CLIENTS = num_client
    data_point = int(len(x_train) / num_client)
    major = int(round(data_point * percent))
    minor = int(data_point - major)

    # Create 100 shards with 480 data points of each label
    shards_480 = [[] for i in range(NUM_SHARDS)]
    for label in range(10):
        label_indices = np.where(y_train == label)[0]
        np.random.shuffle(label_indices)
        num_shards_per_label = NUM_SHARDS // 10
        for i in range(num_shards_per_label):
            shard_indices = label_indices[i * major:(i + 1) * major]
            for j in shard_indices:
                shards_480[label * num_shards_per_label + i].append((x_train[j], label))

    # Create 100 shards with 120 random data points
    shards_120 = [[] for i in range(NUM_SHARDS)]
    all_indices = np.arange(len(y_train))
    for i in range(NUM_SHARDS):
        shard_indices = np.random.choice(all_indices, size=minor, replace=False)
        for j in shard_indices:
            label = y_train[j]
            shards_120[i].append((x_train[j], label))

    # Create 100 clients, each taking one shard from each set of shards
    clients = [[] for i in range(NUM_CLIENTS)]
    for i in range(NUM_CLIENTS):
        clients[i].extend(shards_480[i])
        clients[i].extend(shards_120[i])
        np.random.shuffle(clients[i])

    # Randomly shuffle the clients
    indices = np.random.permutation(NUM_CLIENTS)
    clients = [clients[i] for i in indices]
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_client)]
    return clients, client_names


def mnist_noniid_shuffle_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_non_iid_mnist(x_train, y_train, num_clients, percent=.5,initial='shuffle_clients')


    # Creating bad clients
    clients, bad_client = creating_shuffling_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    # file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_shuffle_mnist.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()
    return sample_list

def mnist_noniid_flip_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_non_iid_mnist(x_train, y_train, num_clients, percent=.5,initial='flip_clients')


    # Creating bad clients
    clients, bad_client = creating_flipping_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    # file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_shuffle_mnist.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()
    return sample_list

def mnist_flip_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')


    # Creating bad clients
    clients, bad_client = creating_flipping_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    # file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_flip_mnist.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()
    return sample_list

def mnist_noniid_noise_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_non_iid_mnist(x_train, y_train, num_clients, percent=.5,initial='noisy_clients')

    # Creating bad clients
    clients, bad_client = creating_noisy_clients_mnist(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    # file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_noise_mnist.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()
    return sample_list

def mnist_noise_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes= 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_client(x_train, y_train, num_clients, initial='client')

    # Creating bad clients
    clients, bad_client = creating_noisy_clients_mnist(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    # file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_noise_mnist.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()
    return sample_list

def create_non_iid_extreme_mnist(x_train, y_train, num_client, percent,initial):
    # Define the number of shards and clients

    NUM_SHARDS = num_client*2
    NUM_CLIENTS = num_client
    data_point = int(len(x_train) / num_client)
    SHARD_SIZE = int(data_point/2)

    # Create 200 shards with unique labels
    shards = [[] for i in range(NUM_SHARDS)]
    for label in range(10):
        label_indices = np.where(y_train == label)[0]
        np.random.shuffle(label_indices)
        num_shards_per_label = NUM_SHARDS // 10
        for i in range(num_shards_per_label):
            shard_indices = label_indices[i * SHARD_SIZE:(i + 1) * SHARD_SIZE]
            for j in shard_indices:
                shards[label * num_shards_per_label + i].append((x_train[j], label))

    # Randomly combine two shards to create 100 clients
    client_shards = [[] for i in range(NUM_CLIENTS)]
    shard_indices = np.arange(NUM_SHARDS)
    np.random.shuffle(shard_indices)
    for i in range(NUM_CLIENTS):
        client_shards[i].extend(shards[shard_indices[i * 2]])
        client_shards[i].extend(shards[shard_indices[i * 2 + 1]])
        np.random.shuffle(client_shards[i])

    # Randomly shuffle the clients
    indices = np.random.permutation(NUM_CLIENTS)
    clients = [client_shards[i] for i in indices]
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_client)]
    return clients, client_names

def cifar10_noniid_extreme_shuffle_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_non_iid_extreme_mnist(x_train, y_train, num_clients, percent=.5,initial='shuffle_clients')


    # Creating bad clients
    clients, bad_client = creating_shuffling_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    # file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_shuffle_mnist.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()
    return sample_list

def cifar10_noniid_extreme_flip_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names = create_non_iid_extreme_mnist(x_train, y_train, num_clients, percent=.5, initial='flip_clients')


    # Creating bad clients
    clients, bad_client = creating_flipping_clients(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    # file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_shuffle_mnist.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()
    return sample_list

def cifar10_noniid_extreme_noise_data(client_percent, data_percent, num_clients):    # Get the data.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    nb_classes = 10
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    dataset, client_names= create_non_iid_extreme_mnist(x_train, y_train, num_clients, percent=.5,initial='noisy_clients')

    # Creating bad clients
    clients, bad_client = creating_noisy_clients_mnist(dataset,client_names,client_percent, data_percent)

    sample_list=[clients, bad_client, x_test, y_test]
    # file_name = f"../../Data/mnist/Dataset{client_percent}_{data_percent}_{num_clients}_noise_mnist.pkl"
    #
    # open_file = open(file_name, "wb")
    # pickle.dump(sample_list, open_file)
    # open_file.close()
    return sample_list