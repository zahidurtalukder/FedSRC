from utils.functions_new import*
from utils.other_algo import *
from utils.math_function import weight_scalling_factor,fed_avg_weight,scale_model_weights,sum_scaled_weights,weight_std_dev_median
import tensorflow as tf
import numpy as np
import copy
import random
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)

import random
import math
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

# Load the Shakespeare dataset
train_data, test_data = tff.simulation.datasets.shakespeare.load_data()

# A fixed vocabularly of ASCII chars that occur in the works of Shakespeare and Dickens:
vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Input pre-processing parameters
SEQ_LENGTH = 100
BATCH_SIZE = 8
BUFFER_SIZE = 100  # For dataset shuffling

# Construct a lookup table to map string chars to indexes,
# using the vocab loaded above:
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab, values=tf.constant(list(range(len(vocab))),
                                       dtype=tf.int64)),
    default_value=0)


def to_ids(x):
  s = tf.reshape(x['snippets'], shape=[1])
  chars = tf.strings.bytes_split(s).values
  ids = table.lookup(chars)
  return ids


def split_input_target(chunk):
  input_text = tf.map_fn(lambda x: x[:-1], chunk)
  target_text = tf.map_fn(lambda x: x[1:], chunk)
  return (input_text, target_text)


def preprocess(dataset):
  return (
      # Map ASCII chars to int64 indexes using the vocab
      dataset.map(to_ids)
      # Split into individual chars
      .unbatch()
      # Form example sequences of SEQ_LENGTH +1
      .batch(SEQ_LENGTH + 1, drop_remainder=True)
      # Shuffle and form minibatches
      .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
      # And finally split into (input, target) tuples,
      # each of length SEQ_LENGTH.
      .map(split_input_target))

class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

  def __init__(self, name='accuracy', dtype=tf.float32):
    super().__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
    return super().update_state(y_true, y_pred, sample_weight)

clients= get_clients()

# Define a function to shuffle 50% of the characters within a text snippet
def shuffle_text(text):
    text_chars = list(text.numpy().decode())  # Convert the text to a list of characters
    num_chars_to_shuffle = len(text_chars) // 1 # Calculate the number of characters to shuffle

    # Randomly shuffle half of the characters
    shuffled_indices = random.sample(range(len(text_chars)), num_chars_to_shuffle)
    for index in shuffled_indices:
        text_chars[index] = random.choice(text_chars)

    shuffled_text = ''.join(text_chars)  # Convert the shuffled characters back to a string
    return shuffled_text

# Use tf.py_function to apply the custom shuffling function to each snippet
def shuffle_snippet(element):
    element['snippets'] = tf.py_function(shuffle_text, [element['snippets']], tf.string)
    return element

def data(client, source=train_data):
  return preprocess(source.create_tf_dataset_for_client(client))

def corrupt_data(client, source=train_data):
    bla=source.create_tf_dataset_for_client(client)
    return preprocess(bla.map(shuffle_snippet))

sla=random.sample(range(0, len(clients)), int(.3*len(clients)))

train_datasets = [corrupt_data(clients[i]) if i in sla else data(clients[i]) for i in range(len(clients))]

# We concatenate the test datasets for evaluation with Keras by creating a
# Dataset of Datasets, and then identity flat mapping across all the examples.
test_dataset = tf.data.Dataset.from_tensor_slices(
    [data(client, test_data) for client in clients]).flat_map(lambda x: x)
# Define the learning rate
lr = 0.001


epochs=150
group_num=5
client_percent =.3

L=1/lr
bla = SimpleMLP5
model = bla.build(1)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[FlattenedCategoricalAccuracy()],
    optimizer=tf.keras.optimizers.Adam(
    learning_rate=lr),
    run_eagerly=True,
    experimental_run_tf_function=False
)
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

    randomlist = random.sample(range(0, len(clients)), math.ceil(len(clients) * client_percent))
    # randomlist= [i for i in range(300)]
    taken_client.append(randomlist)
    total_data = []

    for a in randomlist:
        data_points = 1
        total_data.append(data_points)
        model.set_weights(global_weight)
        local_score = model.evaluate(train_datasets[randomlist[a]], verbose=0)
        model1_accuracy.append(local_score[1])
        model1_loss.append(local_score[0])
        hist1 = model.fit(train_datasets[randomlist[a]], epochs=1, verbose=1)
        weight1 = np.array(model.get_weights())
        model1_train_accuracy.append(hist1.history['accuracy'][-1])
        model1_train_loss.append(hist1.history['loss'][-1])
        model1_weight.append(weight1)


    print(model1_train_loss)

    group1_accuracy.append(model1_accuracy)
    group1_loss.append(model1_loss)
    group1_train_accuracy.append(model1_train_accuracy)
    group1_train_loss.append(model1_train_loss)
    global_weight = median(model1_weight)

    model.set_weights(global_weight)
    score = model.evaluate(test_dataset, verbose=0)
    print('communication round:', i)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    global_accuracy.append(score[1])
    global_loss.append(score[0])

    if i%10==0 and i>0:
        sample_list = [global_accuracy, global_loss, group1_train_accuracy, group1_train_loss, group1_accuracy, group1_loss, global_weight, sla, taken_client]
        save_file_name= f'../../data/noniid/median shakespeare noniid.pkl'
        save_file(save_file_name, sample_list)

