import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L1L2
import numpy as np
# import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, GRU, Dense


class SimpleMLP:
    @staticmethod
    def build(shape, rate):
        model = Sequential()
        model.add(Dense(int(200*rate), input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(int(100*rate), input_shape=(int(200*rate),)))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        return model
    def build_100(shape, classes):
        model = Sequential()
        model.add(Dense(20, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

def save_file(file_name,data):
    open_file = open(file_name, "wb")
    pickle.dump(data, open_file)
    open_file.close()

def open_file(file_name):
    open_file = open(file_name, "rb")
    Dataset = pickle.load(open_file)
    open_file.close()
    return Dataset


def batch_data_non_iid(client_data, client_label, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = client_data,client_label
    label=tf.keras.utils.to_categorical(label, 10)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def batch_data_non_iid_new(client_data, client_label, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    # separate shard into data and labels lists
    id_test = int(len(client_data) * 0.8)
    data, label = client_data[:id_test], client_label[:id_test]
    test_data, test_label = client_data[id_test:-1], client_label[id_test:-1]

    label = tf.keras.utils.to_categorical(label, 10)
    label_test = tf.keras.utils.to_categorical(test_label, 10)  # one-hot encode test labels

    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    test_dataset = tf.data.Dataset.from_tensor_slices((list(test_data), list(label_test)))

    return dataset.shuffle(len(label)).batch(bs), test_dataset.shuffle(len(label_test)).batch(bs)

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

def batch_data_new(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    client_data, client_label = zip(*data_shard)
    id_test = int(len(client_data) * 0.8)
    data, label = client_data[:id_test], client_label[:id_test]
    test_data, test_label = client_data[id_test:-1], client_label[id_test:-1]

    label = tf.keras.utils.to_categorical(label, 10)
    label_test = tf.keras.utils.to_categorical(test_label, 10)  # one-hot encode test labels

    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    test_dataset = tf.data.Dataset.from_tensor_slices((list(test_data), list(label_test)))

    return dataset.shuffle(len(label)).batch(bs), test_dataset.shuffle(len(label_test)).batch(bs)

def batch_data_femnist(data_shard, bs=32):
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

class SimpleMLP3:
    @staticmethod
    def build(shape,rate):
        model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_dim=shape),
        tf.keras.layers.Dense(int(64*rate), activation='relu'),
        tf.keras.layers.Dense(10)
        ])
        return model

class SimpleMLP4:
    @staticmethod
    def build(rate):
        model = Sequential()
        model.add(tf.keras.layers.Conv2D(int(32*rate), (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(32, 32, 3)))
        model.add(tf.keras.layers.Conv2D(int(32*rate), (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(Dense(int(128*rate), activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        return model


def get_submodel(model, percentage_prune):
    r_prune = (1 - percentage_prune)
    m = model
    n = np.array(m) * 0
    for i in range(len(m) - 2):
        if i % 2 == 0:
            x, y = m[i].shape
            y = round(y * r_prune)
            aa = np.transpose(np.array(m[i]))
            bb = np.transpose(n[i])
            bb[:y] = aa[:y]
            cc = np.transpose(bb)
            m[i] = cc
            n[i + 1]
            n[i + 1][:y] = m[i + 1][:y]
            m[i + 1] = n[i + 1]
            n[i + 2]
            n[i + 2][:y] = m[i + 2][:y]
            m[i + 2] = n[i + 2]

    return m

def get_submodel_new(model, node):
    # r_prune = (1 - percentage_prune)
    m = model
    n = np.array(m) * 0
    for i in range(len(m) - 2):
        if i % 2 == 0:
            x, y = m[i].shape
            y = node
            aa = np.transpose(np.array(m[i]))
            bb = np.transpose(n[i])
            bb[:y] = aa[:y]
            cc = np.transpose(bb)
            m[i] = cc
            n[i + 1]
            n[i + 1][:y] = m[i + 1][:y]
            m[i + 1] = n[i + 1]
            n[i + 2]
            n[i + 2][:y] = m[i + 2][:y]
            m[i + 2] = n[i + 2]

    return m

def group_gradient(gradlist, loss_list, group_num, q,qm):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm

    bla=(gm/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla= sum_val**((q-qm)/(qm+1))
    coefm= sla*bla

    gradm=gradlist[0]*0
    for i in range(len(gradlist)):
        gradm= gradm+(pm*(qm+1)*loss_list[i]**qm)*gradlist[i]

    gradm=coefm*gradm
    return gradm

def group_hessian_new(initial_weight, gradlist, loss_list, group_num, q,qm,lr):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm
    L=1/lr

    bla= (gm/(qm+1))*((q-qm)/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla= sum_val**((q-2*qm-1)/(qm+1))
    coefm1= sla*bla

    gradm=gradlist[0]*0
    for i in range(len(loss_list)):
        gradm= gradm+(pm*(qm+1)*loss_list[i]**qm)*gradlist[i]
    grad_val= norm_grad(gradm)/norm_grad(initial_weight)

    bla2= (gm/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla2= sum_val**((q-qm)/(qm+1))
    coefm2= sla2*bla2

    val=0
    for i in range(len(loss_list)):
        val+= pm*(qm+1)*qm*(loss_list[i]**(qm-1))* (norm_grad(gradlist[i])/norm_grad(initial_weight)) + pm*(qm+1)*(loss_list[i]**qm)*L
    hessian= coefm1*grad_val+coefm2*val
    return hessian

def norm_grad(gradm):
    total_grad= []
    for i in range(len(gradm)):
        client_grads = gradm[i].reshape(-1).tolist()
        total_grad+=client_grads
    total_grad= np.array(total_grad)
    return np.sum(np.square(total_grad))



def group_hessian(gradlist, loss_list, group_num, q,qm,lr):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm
    L=1/lr

    bla= (gm/(qm+1))*((q-qm)/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla= sum_val**((q-2*qm-1)/(qm+1))
    coefm1= sla*bla

    gradm=gradlist[0]*0
    for i in range(len(loss_list)):
        gradm= gradm+(pm*(qm+1)*loss_list[i]**qm)*gradlist[i]
    grad_val= norm_grad(gradm)

    bla2= (gm/(qm+1))
    sum_val=0
    for i in range(len(loss_list)):
        sum_val+=pm*(loss_list[i]**(qm+1))
    sla2= sum_val**((q-qm)/(qm+1))
    coefm2= sla2*bla2

    val=0
    for i in range(len(loss_list)):
        val+= pm*(qm+1)*qm*(loss_list[i]**(qm-1))* norm_grad(gradlist[i])+pm*(qm+1) + pm*(qm+1)*(loss_list[i]**qm)*L
    hessian= coefm1*grad_val+coefm2*val
    return hessian

def fed_avg(grad_list):
    gradm= grad_list[0]*0
    for i in range(len(grad_list)):
        gradm= gradm + grad_list[i]
    return gradm/len(grad_list)


def get_submodel_real(large_model,smaller_model):
    m1= large_model
    m2= smaller_model

    for i in range(len(m2)):
        if len(m1[i].shape)==2:
            x1,y1 = m1[i].shape
            x2,y2= m2[i].shape
            if x1==x2 and y1!=y2:
                aa= np.transpose(np.array(m1[i]))
                bb= aa[:y2]

                bb= np.transpose(bb)
                m2[i]=bb
            else:
                bb = m1[i][:x2]
                m2[i]=bb
        else:
            x2= m2[i].shape
            bb= m1[i][:x2[0]]
            m2[i]=bb
    return m2


def get_masked_model(smaller_model, bigger_model):
    m1 = bigger_model
    m2 = smaller_model
    m1 = np.array(m1) * 0
    for i in range(len(m2)):
        if len(m1[i].shape) == 2:
            x1, y1 = m1[i].shape
            x2, y2 = m2[i].shape
            if x1 == x2 and y1 != y2:
                aa = np.transpose(np.array(m1[i]))
                bb = np.transpose(np.array(m2[i]))
                aa[:y2] = bb
                aa = np.transpose(aa)
                m1[i] = aa
            else:
                m1[i][:x2] = m2[i]
        else:
            x2 = m2[i].shape
            m1[i][:x2[0]] = m2[i]
    return m1

def pad_list(small_list, big_list):
    if small_list.shape != big_list.shape:
        pad_widths = [(0, big_list.shape[i] - small_list.shape[i]) for i in range(len(small_list.shape))]
        padded_list = np.pad(small_list, pad_widths, mode='constant')
    else:
        padded_list = small_list
    return padded_list

def group_gradient_chatgpt(gradlist, loss_list, group_num, q,qm):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm

    bla=(gm/(qm+1))
    sum_val = np.sum(np.power(loss_list, qm+1)) * pm
    sla= np.power(sum_val, (q-qm)/(qm+1))
    coefm= sla*bla

    gradm = np.zeros_like(gradlist[0])
    for i in range(len(gradlist)):
        gradm += (pm*(qm+1)*np.power(loss_list[i],qm))*gradlist[i]
    gradm = coefm*gradm
    return gradm

def set_model_weights(model, value):
    weights=[]
    for i in range(len(model)):
        weights.append( np.ones_like(model[i])*value)
    return weights


def get_masked_model_chatgpt(small_model, large_model):
    new_model = []
    for i in range(len(small_model)):
        new_model.append(pad_list(small_model[i], large_model[i]))

    return new_model

def group_hessian_new_chatgpt(initial_weight, gradlist, loss_list, group_num, q,qm ):
    gm=1/group_num
    nm=len(gradlist)
    pm=1/nm
    lr = 0.01
    L=1/lr

    sum_val = np.sum(pm*(np.power(loss_list,qm+1)))
    sla = np.power(sum_val, (q-2*qm-1)/(qm+1))
    coefm1 = (gm/(qm+1))*((q-qm)/(qm+1))*sla

    gradm = np.sum(pm*(qm+1)*np.power(np.array(loss_list).reshape(-1,1),qm)*gradlist, axis=0)
    grad_val = np.linalg.norm(gradm)/np.linalg.norm(initial_weight)

    sum_val = np.sum(pm*(np.power(loss_list,qm+1)))
    sla2 = np.power(sum_val, (q-qm)/(qm+1))
    coefm2 = (gm/(qm+1))*sla2
    val = np.sum(pm*(qm+1)*qm*np.power(loss_list,qm-1)*(np.linalg.norm(gradlist, axis=1)/np.linalg.norm(initial_weight)) + pm*(qm+1)*np.power(loss_list,qm)*L)
    hessian = coefm1*grad_val + coefm2*val

    return hessian

def crop_weight(bigger_weight, smaller_weight):
    """
    Crops the bigger weight to the shape of the smaller weight.
    """
    # Get the shape of the smaller weight
    bigger_weight= np.array(bigger_weight)
    smaller_weight = np.array(smaller_weight)
    smaller_shape = smaller_weight.shape
    # Get the slice indices for each dimension
    slice_indices = [slice(0, sh) for sh in smaller_shape]
    # Crop the bigger weight to the shape of the smaller weight
    cropped_weight = bigger_weight[tuple(slice_indices)]
    return cropped_weight


def get_cropped_model_chatgpt(large_model, small_model):
    new_model = []
    for i in range(len(small_model)):
        new_model.append(crop_weight(large_model[i], small_model[i]))

    return new_model

def get_clients():
    clients=[
        'ALL_S_WELL_THAT_ENDS_WELL_CELIA',
        'MUCH_ADO_ABOUT_NOTHING_OTHELLO',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_BASTARD',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_HOSTESS',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_HUBERT',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_LORD_BARDOLPH',
        'THE_TRAGEDY_OF_KING_LEAR_SHALLOW',
        'THE_TRAGEDY_OF_KING_LEAR_SHYLOCK',
        'THE_TRAGEDY_OF_KING_LEAR_SIMPLE',
        'THE_TRAGEDY_OF_KING_LEAR_SIWARD',
        'THE_TRAGEDY_OF_KING_LEAR_SOLANIO',
        'THE_TRAGEDY_OF_KING_LEAR_THESEUS',
        'THE_TRAGEDY_OF_KING_LEAR_THISBY',
        'THE_TRAGEDY_OF_KING_LEAR_TITANIA',
        'THE_TRAGEDY_OF_KING_LEAR_DUMAIN',
        'ALL_S_WELL_THAT_ENDS_WELL_ADAM',
        'ALL_S_WELL_THAT_ENDS_WELL_AGRIPPA',
        'ALL_S_WELL_THAT_ENDS_WELL_ALEXAS',
        'ALL_S_WELL_THAT_ENDS_WELL_ALL',
        'ALL_S_WELL_THAT_ENDS_WELL_ANTONY',
        'ALL_S_WELL_THAT_ENDS_WELL_ARVIRAGUS',
        'ALL_S_WELL_THAT_ENDS_WELL_AUFIDIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_BELARIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_BRUTUS',
        'ALL_S_WELL_THAT_ENDS_WELL_CAESAR',
        'ALL_S_WELL_THAT_ENDS_WELL_CANIDIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_CELIA',
        'ALL_S_WELL_THAT_ENDS_WELL_CHARLES',
        'ALL_S_WELL_THAT_ENDS_WELL_CHARMIAN',
        'ALL_S_WELL_THAT_ENDS_WELL_CLEOPATRA',
        'ALL_S_WELL_THAT_ENDS_WELL_CLOTEN',
        'ALL_S_WELL_THAT_ENDS_WELL_COMINIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_CORIN',
        'ALL_S_WELL_THAT_ENDS_WELL_CORIOLANUS',
        'ALL_S_WELL_THAT_ENDS_WELL_CORNELIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_CYMBELINE',
        'ALL_S_WELL_THAT_ENDS_WELL_DOLABELLA',
        'ALL_S_WELL_THAT_ENDS_WELL_DUKE_SENIOR',
        'ALL_S_WELL_THAT_ENDS_WELL_ENOBARBUS',
        'ALL_S_WELL_THAT_ENDS_WELL_EROS',
        'ALL_S_WELL_THAT_ENDS_WELL_EUPHRONIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_FIRST_CITIZEN',
        'ALL_S_WELL_THAT_ENDS_WELL_FIRST_GENTLEMAN',
        'ALL_S_WELL_THAT_ENDS_WELL_FIRST_LORD',
        'ALL_S_WELL_THAT_ENDS_WELL_FIRST_SENATOR',
        'ALL_S_WELL_THAT_ENDS_WELL_FIRST_SERVANT',
        'ALL_S_WELL_THAT_ENDS_WELL_FIRST_WATCH',
        'ALL_S_WELL_THAT_ENDS_WELL_FREDERICK',
        'ALL_S_WELL_THAT_ENDS_WELL_GAOLER',
        'ALL_S_WELL_THAT_ENDS_WELL_GUIDERIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_IACHIMO',
        'ALL_S_WELL_THAT_ENDS_WELL_IMOGEN',
        'ALL_S_WELL_THAT_ENDS_WELL_JAQUES',
        'ALL_S_WELL_THAT_ENDS_WELL_LARTIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_LEPIDUS',
        'ALL_S_WELL_THAT_ENDS_WELL_LE_BEAU',
        'ALL_S_WELL_THAT_ENDS_WELL_LUCIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_MAECENAS',
        'ALL_S_WELL_THAT_ENDS_WELL_MARCIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_MENAS',
        'ALL_S_WELL_THAT_ENDS_WELL_MENENIUS',
        'ALL_S_WELL_THAT_ENDS_WELL_MESSENGER',
        'ALL_S_WELL_THAT_ENDS_WELL_OLIVER',
        'ALL_S_WELL_THAT_ENDS_WELL_ORLANDO',
        'PERICLES__PRINCE_OF_TYRE_EXTON',
        'PERICLES__PRINCE_OF_TYRE_FIRST_MURDERER',
        'PERICLES__PRINCE_OF_TYRE_FITZWATER',
        'PERICLES__PRINCE_OF_TYRE_GARDENER',
        'PERICLES__PRINCE_OF_TYRE_GAUNT',
        'PERICLES__PRINCE_OF_TYRE_GHOST',
        'PERICLES__PRINCE_OF_TYRE_GLOUCESTER',
        'PERICLES__PRINCE_OF_TYRE_GREEN',
        'PERICLES__PRINCE_OF_TYRE_HASTINGS',
        'PERICLES__PRINCE_OF_TYRE_KING_EDWARD',
        'PERICLES__PRINCE_OF_TYRE_KING_RICHARD',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_CLIFFORD',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_CONSTABLE',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_CONSTANCE',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_COUNTESS',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_CRANMER',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_CROMWELL',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_DAUPHIN',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_DAVY',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_DICK',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_DOLL',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_DUCHESS',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_EDWARD',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_ELINOR',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_MORTIMER',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_MORTON',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_MOWBRAY',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_NORFOLK',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_NORTHUMBERLAND',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_NYM',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_OLD_LADY',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_ORLEANS',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_OXFORD',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_PAGE',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_PANDULPH',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_PEMBROKE',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_PISTOL',
        'THE_FIRST_PART_OF_KING_HENRY_THE_FOURTH_PLANTAGENET',
        'THE_TAMING_OF_THE_SHREW_ANTONIO',
        'THE_TAMING_OF_THE_SHREW_APEMANTUS',
        'THE_TAMING_OF_THE_SHREW_ARIEL',
        'THE_TAMING_OF_THE_SHREW_BOATSWAIN',
        'THE_TAMING_OF_THE_SHREW_CALIBAN',
        'THE_TAMING_OF_THE_SHREW_FERDINAND',
        'THE_TAMING_OF_THE_SHREW_FIRST_LORD',
        'THE_TAMING_OF_THE_SHREW_FIRST_SENATOR',
        'THE_TAMING_OF_THE_SHREW_FIRST_STRANGER',
        'THE_TAMING_OF_THE_SHREW_FLAVIUS',
        'THE_TRAGEDY_OF_KING_LEAR_ESCALUS',
        'THE_TRAGEDY_OF_KING_LEAR_EVANS',
        'THE_TRAGEDY_OF_KING_LEAR_FAIRY',
        'THE_TRAGEDY_OF_KING_LEAR_FAIRY_QUEEN',
        'THE_TRAGEDY_OF_KING_LEAR_FALSTAFF',
        'THE_TRAGEDY_OF_KING_LEAR_FENTON',
        'THE_TRAGEDY_OF_KING_LEAR_FIRST_GENTLEMAN',
        'THE_TRAGEDY_OF_KING_LEAR_FIRST_WITCH',
        'THE_TRAGEDY_OF_KING_LEAR_FLUTE',
        'THE_TRAGEDY_OF_KING_LEAR_FORD',
        'THE_TRAGEDY_OF_KING_LEAR_GENTLEWOMAN',
        'THE_TRAGEDY_OF_KING_LEAR_GOBBO',
        'THE_TAMING_OF_THE_SHREW_SEMPRONIUS',
        'THE_TAMING_OF_THE_SHREW_SERVANT',
        'THE_TAMING_OF_THE_SHREW_SOLDIER',
        'THE_TAMING_OF_THE_SHREW_STEPHANO',
        'THE_TAMING_OF_THE_SHREW_TIMON',
        'THE_TAMING_OF_THE_SHREW_TITUS',
        'THE_TAMING_OF_THE_SHREW_TRINCULO',
        'THE_TRAGEDY_OF_KING_LEAR_ALL',
        'THE_TAMING_OF_THE_SHREW_FOOL',
        'THE_TAMING_OF_THE_SHREW_GONZALO',
        'THE_TAMING_OF_THE_SHREW_IRIS',
        'THE_TAMING_OF_THE_SHREW_LUCIUS',
        'THE_TAMING_OF_THE_SHREW_MIRANDA',
        'THE_TAMING_OF_THE_SHREW_OLD_ATHENIAN',
        'THE_TAMING_OF_THE_SHREW_PAINTER',
        'PERICLES__PRINCE_OF_TYRE_MARSHAL',
        'PERICLES__PRINCE_OF_TYRE_MESSENGER',
        'PERICLES__PRINCE_OF_TYRE_MOWBRAY',
        'PERICLES__PRINCE_OF_TYRE_NORTHUMBERLAND',
        'PERICLES__PRINCE_OF_TYRE_PERCY',
        'PERICLES__PRINCE_OF_TYRE_PRINCE',
        'PERICLES__PRINCE_OF_TYRE_QUEEN',
        'PERICLES__PRINCE_OF_TYRE_QUEEN_ELIZABETH',
        'PERICLES__PRINCE_OF_TYRE_QUEEN_MARGARET',
        'PERICLES__PRINCE_OF_TYRE_RATCLIFF',
        'PERICLES__PRINCE_OF_TYRE_RICHMOND',
        'PERICLES__PRINCE_OF_TYRE_RIVERS'

    ]
    return clients

class SimpleMLP5:
    @staticmethod
    def build(rate):
        model = Sequential()
        model.add(Embedding(86, 256, batch_input_shape=[8, None]))
        model.add(GRU(int(1024*rate), return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
        model.add(Dense(86))
        return model

def RHI(unique_labels, counts, max_class=10, gamma=0.1):
    sum_count = np.sum(counts)
    hi= 1-((len(unique_labels)-1)/(max_class-1))
    bla = 0
    for i in range(len(counts)):
        prob = counts[i] / sum_count
        if prob > 0:  # Avoid log(0)
            bla += prob * np.log(prob)
    ge=(-bla)/np.log(len(counts))
    
    return gamma * hi + (1-gamma) * (1-ge)

def RHI_cal(data_shard, max_class,gamma):
    client_data, labels = zip(*data_shard)
    unique_labels, counts = np.unique(labels, return_counts=True)
    rhi_value = RHI(unique_labels, counts, max_class, gamma)
    # Check if the result is NaN, and return 0 if it is
    if np.isnan(rhi_value):
        return 0
    return rhi_value
