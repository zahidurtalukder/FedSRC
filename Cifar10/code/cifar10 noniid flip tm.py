from utils.functions_new import *
from utils.cifar10_data_generator import *
from utils.other_algo import *
from utils.math_function import weight_scalling_factor,fed_avg_weight,scale_model_weights,sum_scaled_weights,weight_std_dev_median
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras import backend as K
import random
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)

# # file_name = "../Dataset100_clean_mnist.pkl"
# Dataset= open_file(file_name)
Dataset= cifar10_noniid_flip_data(client_percent=.3, data_percent=1,num_clients=100)
bad_client_shuffle= Dataset[1]
#process and batch the training data for each client
clients= Dataset[0]
clients_batched = dict()
clients_batched_test = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name],clients_batched_test[client_name]= batch_data_new(data)

#process and batch the test set
bad_client= Dataset[1]
x_test= Dataset[2]
y_test= Dataset[3]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
client_names = list(clients_batched.keys())


loss = 'categorical_crossentropy'
metrics = ['accuracy']
epochs = 300
lr = 0.0001
alpha= 1
beta=1
batch_size = 32
client_percent= .5
bla = SimpleMLP4
model = bla.build(1)
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
global_mean=0
global_standard_dev=0
bla=0

for i in range(epochs):
    print("group 1 training")
    model1_accuracy = []
    model1_loss = []
    model1_train_accuracy = []
    model1_train_loss = []
    model1_weight = []
    fileter1_block=[]
    fileter2_block=[]

    # randomlist = random.sample(range(0, 300), math.ceil(300 * client_percent))
    randomlist= [i for i in range(100)]
    taken_client.append(randomlist)
    total_data = []

    for a in randomlist:
        data_points = len(clients_batched[client_names[a]]) * batch_size
        total_data.append(data_points)
        model.set_weights(global_weight)
        local_score = model.evaluate(clients_batched[client_names[a]], verbose=0)
        # if i % 1 == 0 and i > 0:
        #     score1 = model.evaluate(clients_batched_test[client_names[a]], verbose=1)
        #     model1_accuracy.append(score1[1])
        #     model1_loss.append(score1[0])
        model1_accuracy.append(local_score[1])
        model1_loss.append(local_score[0])
        hist1 = model.fit(clients_batched[client_names[a]], epochs=1, verbose=1)
        weight1 = np.array(model.get_weights())
        model1_train_accuracy.append(hist1.history['accuracy'][-1])
        model1_train_loss.append(hist1.history['loss'][-1])
        model1_weight.append(weight1)
        K.clear_session()



    print(model1_train_loss)


    group1_accuracy.append(model1_accuracy)
    group1_loss.append(model1_loss)
    group1_train_accuracy.append(model1_train_accuracy)
    group1_train_loss.append(model1_train_loss)
    global_weight = trimmed_mean(model1_weight, .2)

    model.set_weights(global_weight)
    model.evaluate(x_test, y_test)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('communication round:', i)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    global_accuracy.append(score[1])
    global_loss.append(score[0])

    if i%10==0 and i>0:
        global_weight_list.append(global_weight)
        sample_list = [global_accuracy, global_loss, group1_train_accuracy, group1_train_loss, group1_accuracy, group1_loss, global_weight_list,bad_client_shuffle]
        save_file_name= f'../data/cifar10 tm flip.pkl'
        save_file(save_file_name, sample_list)

