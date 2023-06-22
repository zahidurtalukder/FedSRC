# implementing FL
import numpy as np
import random
import keras
import tensorflow as tf

def create_client(image_list, label_list, num_clients,initial='clients'):
    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))
    return shards, client_names


def creating_shuffling_clients(dataset, client_names, client_percent, data_percent):
    print('this is shuffling attack')
    shards = dataset
    num_clients = len(client_names)
    random_client = random.sample(range(0, num_clients), round(num_clients * client_percent))
    random_client = np.sort(random_client)
    if len(random_client) == 0:
        print("All_ good_CLients")
    else:
        print(random_client)
        x = 0
        y = 0
        for i in range(num_clients):
            if random_client[y] == x:
                print("engaged: ", i)
                data, label = zip(*shards[i])
                label = np.array(label)
                data_length = len(label)
                random_data = random.sample(range(0, data_length), round(data_length * data_percent))
                random_data = np.sort(random_data)
                m = 0
                n = 0

                for a in range(data_length):
                    if random_data[n] == m:
                        label[a] = random.randrange(0, 9, 1)
                        if n < len(random_data) - 1:
                            n = n + 1
                    m = m + 1
                print(label)
                new_shards = list(zip(data, label))
                shards[i] = new_shards

                if y < len(random_client) - 1:
                    y = y + 1
            x = x + 1

    return {client_names[i]: shards[i] for i in range(len(client_names))}, random_client

def creating_flipping_clients(dataset, client_names, client_percent, data_percent):
    print('this is fipping attack')
    shards = dataset
    num_clients = len(client_names)
    random_client = random.sample(range(0, num_clients), round(num_clients * client_percent))
    random_client = np.sort(random_client)
    if len(random_client) == 0:
        print("All_ good_CLients")
    else:
        print(random_client)
        x = 0
        y = 0
        for i in range(num_clients):
            if random_client[y] == x:
                print("engaged: ", i)
                data, label = zip(*shards[i])
                label = np.array(label)
                data_length = len(label)
                random_data = random.sample(range(0, data_length), round(data_length * data_percent))
                random_data = np.sort(random_data)
                m = 0
                n = 0
                fake_label= random.randrange(0,9,1)
                print(fake_label)
                for a in range(data_length):
                    if random_data[n] == m:
                        label[a] = fake_label
                        if n < len(random_data) - 1:
                            n = n + 1
                    m = m + 1
                print(label)
                new_shards = list(zip(data, label))
                shards[i] = new_shards

                if y < len(random_client) - 1:
                    y = y + 1
            x = x + 1

    return {client_names[i]: shards[i] for i in range(len(client_names))}, random_client

def creating_noisy_clients_mnist(dataset, client_names, client_percent, data_percent):
    print('this is fipping attack')
    shards = dataset
    num_clients = len(client_names)
    random_client = random.sample(range(0, num_clients), round(num_clients * client_percent))
    random_client = np.sort(random_client)
    if len(random_client) == 0:
        print("All_ good_CLients")
    else:
        print(random_client)
        x = 0
        y = 0
        for i in range(num_clients):
            if random_client[y] == x:
                print("engaged: ", i)
                data, label = zip(*shards[i])
                label = np.array(label)
                data= np.array(data)
                data_length = len(label)
                random_data = random.sample(range(0, data_length), round(data_length * data_percent))
                random_data = np.sort(random_data)
                m = 0
                n = 0
                for a in range(data_length):
                    if random_data[n] == m:
                        mu, sigma = 0, 0.7
                        noise = np.random.normal(mu, sigma, [1, 784])
                        data[a] = data[a] + noise
                        data[a] = (data[a] - np.min(data[a])) / (np.max(data[a]) - np.min(data[a]))
                        if n < len(random_data) - 1:
                            n = n + 1
                    m = m + 1

                new_shards = list(zip(data, label))
                shards[i] = new_shards

                if y < len(random_client) - 1:
                    y = y + 1
            x = x + 1

    return {client_names[i]: shards[i] for i in range(len(client_names))}, random_client


def creating_noisy_clients_cifar10(dataset, client_names, client_percent, data_percent):
    print('this is fipping attack')
    shards = dataset
    num_clients = len(client_names)
    random_client = random.sample(range(0, num_clients), round(num_clients * client_percent))
    random_client = np.sort(random_client)
    if len(random_client) == 0:
        print("All_ good_CLients")
    else:
        print(random_client)
        x = 0
        y = 0
        for i in range(num_clients):
            if random_client[y] == x:
                print("engaged: ", i)
                data, label = zip(*shards[i])
                label = np.array(label)
                data = np.array(data)
                data_length = len(label)
                random_data = random.sample(range(0, data_length), round(data_length * data_percent))
                random_data = np.sort(random_data)
                m = 0
                n = 0
                for a in range(data_length):
                    if random_data[n] == m:
                        mu, sigma = 0, 0.7
                        noise = np.random.normal(mu, sigma, [32,32,3])
                        data[a] = data[a] + noise
                        data[a] = (data[a] - np.min(data[a])) / (np.max(data[a]) - np.min(data[a]))
                        if n < len(random_data) - 1:
                            n = n + 1
                    m = m + 1

                new_shards = list(zip(data, label))
                shards[i] = new_shards

                if y < len(random_client) - 1:
                    y = y + 1
            x = x + 1

    return {client_names[i]: shards[i] for i in range(len(client_names))}, random_client

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    label=tf.keras.utils.to_categorical(label, 10)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def batch_data_femnist(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def creating_targetted_clients(dataset, client_names, client_percent, data_percent):
    print('this is targetted attack')
    shards = dataset
    num_clients = len(client_names)
    random_client = random.sample(range(0, num_clients), round(num_clients * client_percent))
    random_client = np.sort(random_client)
    if len(random_client) == 0:
        print("All_ good_CLients")
    else:
        print(random_client)
        x = 0
        y = 0
        for i in range(num_clients):
            if random_client[y] == x:
                print("engaged: ", i)
                data, label = zip(*shards[i])
                label = np.array(label)
                data_length = len(label)
                random_data = random.sample(range(0, data_length), round(data_length * data_percent))
                random_data = np.sort(random_data)
                m = 0
                n = 0

                for a in range(data_length):
                    if random_data[n] == m:
                        # label[a] = random.randrange(0, 9, 1)
                        if label[a]==0:
                            label[a]=9
                        elif label[a]==1:
                            label[a]=8
                        elif label[a]==2:
                            label[a]=7
                        elif label[a]==3:
                            label[a]=6
                        elif label[a]==4:
                            label[a]=5
                        elif label[a]==5:
                            label[a]=4
                        elif label[a]==6:
                            label[a]=3
                        elif label[a]==7:
                            label[a]=2
                        elif label[a]==8:
                            label[a]=1
                        elif label[a]==9:
                            label[a]=0

                        if n < len(random_data) - 1:
                            n = n + 1
                    m = m + 1
                print(label)
                new_shards = list(zip(data, label))
                shards[i] = new_shards

                if y < len(random_client) - 1:
                    y = y + 1
            x = x + 1

    return {client_names[i]: shards[i] for i in range(len(client_names))}, random_client

def creating_targetted_shuffling_clients(dataset, client_names, client_percent, data_percent,percent):
    print('this is targetted attack')
    shards = dataset
    num_clients = len(client_names)
    random_client = random.sample(range(0, num_clients), round(num_clients * client_percent))
    bla= random.sample(random_client, percent)
    random_client = np.sort(random_client)
    if len(random_client) == 0:
        print("All_ good_CLients")
    else:
        print(random_client)
        x = 0
        y = 0
        for i in range(num_clients):
            if random_client[y] == x:
                print("engaged: ", i)
                data, label = zip(*shards[i])
                label = np.array(label)
                data_length = len(label)
                random_data = random.sample(range(0, data_length), round(data_length * data_percent))
                random_data = np.sort(random_data)
                m = 0
                n = 0
                if random_client[y] in bla:
                    print('target',random_client[y])
                    for a in range(data_length):
                        # if random_data[n] == m:
                        label[a] = random.randrange(0, 9, 1)
                            # if n < len(random_data) - 1:
                            #     n = n + 1
                else:



                    for a in range(data_length):
                        # if random_data[n] == m:
                            # label[a] = random.randrange(0, 9, 1)
                        if label[a]==0:
                            label[a]=9
                        elif label[a]==1:
                            label[a]=8
                        elif label[a]==2:
                            label[a]=7
                        elif label[a]==3:
                            label[a]=6
                        elif label[a]==4:
                            label[a]=5
                        elif label[a]==5:
                            label[a]=4
                        elif label[a]==6:
                            label[a]=3
                        elif label[a]==7:
                            label[a]=2
                        elif label[a]==8:
                            label[a]=1
                        elif label[a]==9:
                            label[a]=0

                            # if n < len(random_data) - 1:
                            #     n = n + 1
                    m = m + 1
                # print(label)
                new_shards = list(zip(data, label))
                shards[i] = new_shards

                if y < len(random_client) - 1:
                    y = y + 1
            x = x + 1

    return {client_names[i]: shards[i] for i in range(len(client_names))}, random_client

def creating_targetted_flipping_clients(dataset, client_names, client_percent, data_percent,percent):
    print('this is targetted attack')
    shards = dataset
    num_clients = len(client_names)
    random_client = random.sample(range(0, num_clients), round(num_clients * client_percent))
    bla= random.sample(random_client, percent)
    random_client = np.sort(random_client)
    if len(random_client) == 0:
        print("All_ good_CLients")
    else:
        print(random_client)
        x = 0
        y = 0
        for i in range(num_clients):
            if random_client[y] == x:
                print("engaged: ", i)
                data, label = zip(*shards[i])
                label = np.array(label)
                data_length = len(label)
                random_data = random.sample(range(0, data_length), round(data_length * data_percent))
                random_data = np.sort(random_data)
                m = 0
                n = 0
                fake_label = random.randrange(0, 9, 1)
                if random_client[y] in bla:
                    print('target',random_client[y])
                    for a in range(data_length):
                        # if random_data[n] == m:
                        label[a] = 5
                            # if n < len(random_data) - 1:
                            #     n = n + 1
                else:



                    for a in range(data_length):
                        # if random_data[n] == m:
                        label[a] = fake_label

                            # if n < len(random_data) - 1:
                            #     n = n + 1
                    m = m + 1
                # print(label)
                new_shards = list(zip(data, label))
                shards[i] = new_shards

                if y < len(random_client) - 1:
                    y = y + 1
            x = x + 1

    return {client_names[i]: shards[i] for i in range(len(client_names))}, random_client