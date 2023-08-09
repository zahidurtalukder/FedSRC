from utils.functions_new import *
from utils.other_algo import *
from utils.math_function import weight_scalling_factor,fed_avg_weight,scale_model_weights,sum_scaled_weights,weight_std_dev_median
import tensorflow as tf
import numpy as np
from utils.client_creation import *
import math
# from tensorflow.keras import backend as K
import random
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)

num_clients=200

file_name = "../../Dataset_clean_3000.pkl"
Dataset= open_file(file_name)
keys_list = list(Dataset[0].keys())
values_list = list(Dataset[0].values())
clients, bad_client_shuffle = creating_shuffling_clients(values_list[:num_clients],keys_list[:num_clients],client_percent=.3, data_percent=1)
clients2, bad_client_flip = creating_flipping_clients(values_list[num_clients:num_clients*2],keys_list[num_clients:num_clients*2],client_percent=.3, data_percent=1)
clients3, bad_client_noise = creating_noisy_clients_mnist(values_list[num_clients*2:num_clients*3],keys_list[num_clients*2:num_clients*3],client_percent=.3, data_percent=1)

clients.update(clients2)
clients.update(clients3)
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data_femnist(data)

file_name = "../../Testset_3000.pkl"
Testset_bla= open_file(file_name)
#process and batch the test set
x_test= Testset_bla[0]
y_test= Testset_bla[1]
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
client_names = list(clients_batched.keys())

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']
epochs = 300
lr = 0.001
alpha= 1
beta=1
batch_size = 32
client_percent= .3
bla = SimpleMLP3
model = bla.build(784,1)
model.compile(loss=loss,
              optimizer=tf.keras.optimizers.Adam(
                  learning_rate=lr),
              metrics=metrics)
global_weight = model.get_weights()
initial_weight = model.get_weights()


group1_accuracy = []
group1_loss = []
group1_train_accuracy=[]
group1_train_loss=[]
global_accuracy = []
global_loss = []
global_weight_list=[]
taken_client = []


for i in range(epochs):
    print("group 1 training")
    model1_accuracy = []
    model1_loss = []
    model1_train_accuracy = []
    model1_train_loss = []
    model1_weight = []
    fileter1_block=[]
    fileter2_block=[]

    randomlist = random.sample(range(0, num_clients*3), math.ceil(num_clients*3* client_percent))
    # randomlist= [i for i in range(300)]
    taken_client.append(randomlist)
    total_data = []

    for a in randomlist:
        data_points = len(clients_batched[client_names[a]]) * batch_size
        total_data.append(data_points)
        model.set_weights(global_weight)
        local_score = model.evaluate(clients_batched[client_names[a]], verbose=0)
        model1_accuracy.append(local_score[1])
        model1_loss.append(local_score[0])
        hist1 = model.fit(clients_batched[client_names[a]], epochs=1, verbose=1)
        weight1 = np.array(model.get_weights())
        model1_train_accuracy.append(hist1.history['accuracy'][-1])
        model1_train_loss.append(hist1.history['loss'][-1])
        model1_weight.append(weight1)


    print(model1_train_loss)


    group1_accuracy.append(model1_accuracy)
    group1_loss.append(model1_loss)
    group1_train_accuracy.append(model1_train_accuracy)
    group1_train_loss.append(model1_train_loss)
    global_weight = krum_new(model1_weight,.3)

    model.set_weights(global_weight)
    model.evaluate(x_test, y_test)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('communication round:', i)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    global_accuracy.append(score[1])
    global_loss.append(score[0])

    if i%10==0 and i>0:
        sample_list = [global_accuracy, global_loss, group1_train_accuracy, group1_train_loss, group1_accuracy, group1_loss, global_weight, bad_client_flip, bad_client_shuffle, bad_client_flip, taken_client]
        save_file_name= f'../../data/noniid/krum femnist noniid.pkl'
        save_file(save_file_name, sample_list)

