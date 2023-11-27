import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def one_hot(x):
    y = np.zeros((x.max() + 1, x.size))
    y[x, np.arange(x.size)] = 1
    return y

def check_label(x, u):
    '''if x=u return 1, else return 0'''
    return np.where(x == u,1,0)



def random_false_positive_index(labels, data_y):
    false_positive_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 1))
    if false_positive_indices.size == 0:
        return -1
    return np.random.choice(false_positive_indices, 1, replace=True)

#union of convex separators initialise
def hyperplane_init_objective(w, data_x, data_y, x_fp, set_indices):
    dot_prod = np.dot(w, data_x) - np.dot(w, x_fp)
    dot_prod_1 = np.copy(dot_prod)
    dot_prod_2 = np.copy(dot_prod)
    dot_prod_1[dot_prod_1 >= 1] = 1
    dot_prod_2[dot_prod_2 >= 0] = 0
    return np.sum((dot_prod_1 + dot_prod_2) * check_label(data_y[set_indices], 1))

def append_bias(data_x):
    data_x_temp = np.zeros((data_x.shape[0]+1, data_x.shape[1]))
    data_x_temp[0:-1, :] = data_x
    data_x_temp[-1, :] = np.ones(data_x.shape[1])
    return data_x_temp

'''
#approach1
def hyperplane_init_objective(w, data_x, data_y, x_fp):
    dot_prod = np.dot(w, data_x) - np.dot(w, x_fp)
    dot_prod[dot_prod>=1] = 1
    return np.sum(dot_prod * check_label(data_y, 1))
'''