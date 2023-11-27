import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os
from fpdf import FPDF
from helpers import *
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from collections import Counter


color_map = {}
dir_points = {}
j=0
i=0

for k in range(0, 150):
    color_map[k] = cm.tab20(k)
    if i*0.2 == 1.8:
        j += 0.1
        i=0
    dir_points[k] = [0.8 + j, i*0.2]
    i+=1

def converged(W_1,W_2):
    diff_theta=W_1-W_2
    diff_theta=abs(diff_theta)
    if ((diff_theta<1e-6).all()):
        return True
    return False

def add_hyperplane(W, S, data_x, data_y, first, c, pdf, visualise = False):
    not_in_the_hull = False
    while(~not_in_the_hull):
        w, not_in_the_hull, separator_index, x, fp_index = learn_init_hyperplane(W, S, data_x, data_y, first, pdf, visualise)
        added_hyperplane_indices = []
        if w is None:
            return W, [], data_x, data_y
        if first:
            w = w
            if not_in_the_hull:
                W.append(w.reshape(-1, 1))
                added_hyperplane_indices.append([0,0])
            else:
                w[-1] = -1 * np.dot(w[0:-1], x[0:-1])
                data_s = data_x[:, np.where(S == separator_index)[0]]

                #finding random hyperplane by finding max of min absolute distance from the positive points
                tries = 5
                min_dist = np.zeros(tries)
                w_s = []
                for i in range(0, tries):
                    w = -1 + 2 * np.random.random(data_x.shape[0])
                    w[-1] = -1 * np.dot(w[0:-1], x[0:-1])
                    w_s.append(w)
                    data_pos = data_x[:, np.where(data_y == 1)[0]]
                    min_dist[i] = np.amin(np.abs(np.dot(w, data_pos))/ np.linalg.norm(w[0:-1]))
                w = w_s[np.argmax(min_dist)]

                H_1 = [w]
                H_2 = [-1 * w]
                H_1 = np.asarray(H_1).T
                H_2 = np.asarray(H_2).T
                W.append(H_1)
                W.append(H_2)
                added_hyperplane_indices.append([0, 0])
                added_hyperplane_indices.append([1, 0])
            return W, added_hyperplane_indices, data_x, data_y

        if not_in_the_hull:
            w_new = np.zeros((W[separator_index].shape[0], W[separator_index].shape[1] + 1))
            scale = np.amin(np.abs(W[separator_index])) + np.random.random() * (np.amax(np.abs(W[separator_index])) - np.amin(np.abs(W[separator_index])))
            w_new[:,0:-1] = W[separator_index]
            w_new[:,-1] = (w/np.linalg.norm(w)) * scale
            W[separator_index] = w_new
            added_hyperplane_indices.append([separator_index, W[separator_index].shape[1]-1])
            return W, added_hyperplane_indices, data_x, data_y

        if len(W) <= c:
            #data_s = data_x[:, np.intersect1d(np.where(S == separator_index)[0], np.where(data_y == 1)[0])]
            data_s = data_x[:, np.where(S == separator_index)[0]]
            w[-1] = -1 * np.dot(w[0:-1], x[0:-1])
            # finding random hyperplane by finding max of min absolute distance from the positive points
            tries = 5
            min_dist = np.zeros(tries)
            w_s = []
            for i in range(0, tries):
                w = -1 + 2 * np.random.random(data_x.shape[0])
                w[-1] = -1 * np.dot(w[0:-1], x[0:-1])
                w_s.append(w)
                data_pos = data_x[:, np.where(data_y == 1)[0]]
                min_dist[i] = np.amin(np.abs(np.dot(w, data_pos)) / np.linalg.norm(w[0:-1]))
            w = w_s[np.argmax(min_dist)]

            #finding range of max min distribution
            max_coeff = 0
            min_coeff = 0
            for i in range(len(W)):
                max_coeff = np.amax(np.abs(W[i]))
                min_coeff = np.amin(np.abs(W[i]))
            scale = min_coeff + np.random.random()*(max_coeff - min_coeff)
            S1 = np.where(np.dot(w, data_s) > 0)[0]
            S2 = np.where(np.dot(w, data_s) < 0)[0]
            w = (w/np.linalg.norm(w))*scale
            H_1 = [w]
            H_2 = [-1*w]
            for i in range(0, W[separator_index].shape[1]):
                hyp = W[separator_index][:, i]
                m1 = np.amin(np.dot(hyp, data_s[:, S1]))
                m2 = np.amin(np.dot(hyp, data_s[:, S2]))
                if m1 < m2:
                    H_1.append(hyp)
                else:
                    H_2.append(hyp)
            H_1 = np.asarray(H_1).T
            H_2 = np.asarray(H_2).T

            W[separator_index] = H_1
            W.insert(separator_index+1, H_2)
            added_hyperplane_indices.append([separator_index, 0])
            added_hyperplane_indices.append([separator_index+1, 0])

            return W, added_hyperplane_indices, data_x, data_y
        data_x = np.delete(data_x, fp_index, 1)
        data_y = np.delete(data_y, fp_index)

def add_hyperplane_1(W, S, data_x, data_y, first, c, pdf, visualise=False):
    w, not_in_the_hull, separator_index, x, _ = learn_init_hyperplane(W, S, data_x, data_y, first, pdf, visualise)
    added_hyperplane_indices = []
    if w is None:
        return W, []
    if first:
        w = w
        if not_in_the_hull:
            W.append(w.reshape(-1, 1))
            added_hyperplane_indices.append([0, 0])
        else:
            w[-1] = -1 * np.dot(w[0:-1], x[0:-1])
            data_s = data_x[:, np.where(S == separator_index)[0]]

            # finding random hyperplane by finding max of min absolute distance from the positive points
            tries = 5
            min_dist = np.zeros(tries)
            w_s = []
            for i in range(0, tries):
                w = -1 + 2 * np.random.random(data_x.shape[0])
                w[-1] = -1 * np.dot(w[0:-1], x[0:-1])
                w_s.append(w)
                data_pos = data_x[:, np.where(data_y == 1)[0]]
                min_dist[i] = np.amin(np.abs(np.dot(w, data_pos)) / np.linalg.norm(w[0:-1]))
            w = w_s[np.argmax(min_dist)]

            H_1 = [w]
            H_2 = [-1 * w]
            H_1 = np.asarray(H_1).T
            H_2 = np.asarray(H_2).T
            W.append(H_1)
            W.append(H_2)
            added_hyperplane_indices.append([0, 0])
            added_hyperplane_indices.append([1, 0])
        return W, added_hyperplane_indices

    if not_in_the_hull:
        if W[separator_index].shape[1] >= c:
            return W, added_hyperplane_indices
        w_new = np.zeros((W[separator_index].shape[0], W[separator_index].shape[1] + 1))
        scale = np.amin(np.abs(W[separator_index])) + np.random.random() * (
                    np.amax(np.abs(W[separator_index])) - np.amin(np.abs(W[separator_index])))
        w_new[:, 0:-1] = W[separator_index]
        w_new[:, -1] = (w / np.linalg.norm(w)) * scale
        W[separator_index] = w_new
        added_hyperplane_indices.append([separator_index, W[separator_index].shape[1] - 1])
        return W, added_hyperplane_indices

    # data_s = data_x[:, np.intersect1d(np.where(S == separator_index)[0], np.where(data_y == 1)[0])]
    data_s = data_x[:, np.where(S == separator_index)[0]]
    w[-1] = -1 * np.dot(w[0:-1], x[0:-1])
    # finding random hyperplane by finding max of min absolute distance from the positive points
    tries = 5
    min_dist = np.zeros(tries)
    w_s = []
    for i in range(0, tries):
        w = -1 + 2 * np.random.random(data_x.shape[0])
        w[-1] = -1 * np.dot(w[0:-1], x[0:-1])
        w_s.append(w)
        data_pos = data_x[:, np.where(data_y == 1)[0]]
        min_dist[i] = np.amin(np.abs(np.dot(w, data_pos)) / np.linalg.norm(w[0:-1]))
    w = w_s[np.argmax(min_dist)]

    # finding range of max min distribution
    max_coeff = 0
    min_coeff = 0
    for i in range(len(W)):
        max_coeff = np.amax(np.abs(W[i]))
        min_coeff = np.amin(np.abs(W[i]))
    scale = min_coeff + np.random.random() * (max_coeff - min_coeff)
    S1 = np.where(np.dot(w, data_s) > 0)[0]
    S2 = np.where(np.dot(w, data_s) < 0)[0]
    w = (w / np.linalg.norm(w)) * scale
    H_1 = [w]
    H_2 = [-1 * w]
    for i in range(0, W[separator_index].shape[1]):
        hyp = W[separator_index][:, i]
        m1 = np.amin(np.dot(hyp, data_s[:, S1]))
        m2 = np.amin(np.dot(hyp, data_s[:, S2]))
        if m1 < m2:
            H_1.append(hyp)
        else:
            H_2.append(hyp)
    H_1 = np.asarray(H_1).T
    H_2 = np.asarray(H_2).T

    W[separator_index] = H_1
    W.insert(separator_index + 1, H_2)
    added_hyperplane_indices.append([separator_index, 0])
    added_hyperplane_indices.append([separator_index + 1, 0])

    return W, added_hyperplane_indices

def remove_dead_hyperplanes(W, indices):
    if len(indices)==0:
        return W
    if W.shape[1] == len(indices):
        return None
    W_new = np.zeros((W.shape[0], W.shape[1] - len(indices)))

    k=0
    for i in range(W.shape[1]):
        if i not in indices:
            W_new[:, k] = W[:, i]
            k+=1
    return W_new

def indicator_min(x):
    return (x <= np.sort(x, axis=0)[::-1][[-1], :]).astype(int)

def indicator_max(x):
    return (x >= np.sort(x, axis=0)[[-1], :]).astype(int)

def refresh_memberships(W, data_x):
    y_i_hat = []
    for i in range(0, len(W)):
        #y_i_hat.append(np.amin(np.dot((W[i]/np.linalg.norm(W[i][0:-1, :], axis=0)).T, data_x), axis=0))
        y_i_hat.append(np.amin(np.dot(W[i].T, data_x), axis=0))
    y_i_hat = np.asarray(y_i_hat)
    S = np.argmax(y_i_hat, axis=0)
    return S

def y_i_hat(W, S, data_x, return_intermediates = 0):
    y_i_hat = []
    intermediate = []
    for i in range(0, len(W)):
        temp = np.dot(W[i].T, data_x)
        y_i_hat.append(np.amin(temp, axis=0))
        intermediate.append(temp)
    y_i_hat = np.asarray(y_i_hat)
    y_i_hat = y_i_hat[S, np.arange(0, data_x.shape[1])]
    if return_intermediates!=1:
        return y_i_hat
    return y_i_hat, intermediate

def logistic(x):
    return 1/(1+np.exp(-1*x))

def h_w(W, S, data_x, return_intermediates = 0):
    if return_intermediates!=1:
        temp = y_i_hat(W, S, data_x)
        return logistic(temp)
    temp1, intermediate = y_i_hat(W, S, data_x, return_intermediates)
    return logistic(temp1), intermediate

def l_theta(W, S, data_x, data_y, return_intermediates = 0):
    if return_intermediates != 1:
        return np.sum(data_y*np.log(h_w(W, S, data_x)) + (1-data_y)*np.log(1-h_w(W, S, data_x)))
    h, intermediate = h_w(W, S, data_x, return_intermediates)
    return np.sum(data_y*np.log(h) + (1-data_y)*np.log(1-h)), intermediate

def gradient(W, S, data_x, data_y, dot_prods):
    max_grps = np.zeros((len(W), data_x.shape[1]), dtype = 'int')
    max_grps[S, np.arange(0, data_x.shape[1])] = 1
    total_derivative = []
    h = h_w(W, S, data_x)
    for i in range(0, len(W)):
        #num_hyp_grp = np.where(max_grp_index == i)[0]
        #if len(num_hyp_grp) == 0:
        #    continue
        derivative = np.zeros_like(W[i])
        temp = dot_prods[i]
        indicator_pos = indicator_min(temp)
        min_hyp_indices = np.where(indicator_pos == 1)[0]
        min_x_indices = np.where(indicator_pos == 1)[1]
        for j in range(0, W[i].shape[1]):
            num_hyp = np.where(min_hyp_indices == j)[0]
            if len(num_hyp) == 0:
                continue
            x_i_for_w_i = data_x[:, min_x_indices[num_hyp]]
            if (len(x_i_for_w_i.shape) == 1):
                derivative[:, j] += (data_y - h)[min_x_indices[num_hyp]] * max_grps[i, min_x_indices[num_hyp]] * x_i_for_w_i
            else:
                derivative[:, j] += np.sum(
                    np.tile((data_y - h)[min_x_indices[num_hyp]] * max_grps[i, min_x_indices[num_hyp]], (x_i_for_w_i.shape[0], 1)) * x_i_for_w_i, axis=1)
        total_derivative.append(derivative)
    return total_derivative

def update_w(W, derivative, learn_rate):
    W_new = []
    for i in range(0, len(W)):
        W_new.append(W[i] + learn_rate*derivative[i])
    return W_new

def reset_removal_counts(W):
    count_zero_derivative = []
    count_zero_one_sided = []
    for i in range(0, len(W)):
        count_zero_derivative.append(np.zeros(W[i].shape[1]))
        count_zero_one_sided.append(np.zeros(W[i].shape[1]))
    return count_zero_derivative, count_zero_one_sided

def learn_init_hyperplane(W, S, data_x, data_y, first, pdf, visualise = False):
    w = -1 + 2 * np.random.random(data_x.shape[0])
    if first:
        x_fp = np.sum(data_x[:, np.where(data_y == 0)[0]], axis=1)/np.sum(np.where(data_y == 0, 1, 0))
        fp_index = 0
    else:
        labels = predict(W, data_x)
        fp_index = random_false_positive_index(labels, data_y)
        if fp_index == -1:
            #see the count of false negatives
            false_negative_indices = np.intersect1d(np.where(labels == 0), np.where(data_y == 1))
            if len(false_negative_indices)/np.sum(np.where(data_y==1,1,0))>0.01:
                x_fp = np.sum(data_x[:, np.where(data_y == 0)[0]], axis=1) / np.sum(np.where(data_y == 0, 1, 0))
                fp_index = 0
            else:
                #The model has already learnt
                return None, None, None, None, None
        else:
            false_positive_indices = np.intersect1d(np.where(labels == 1), np.where(data_y == 0))
            if (len(false_positive_indices) / len(data_y)) > 0.0001: #there is still improvement to be done
                fp_index = fp_index[0]
                x_fp = data_x[:, fp_index]
                positive_indices = np.intersect1d(np.where(S == S[fp_index])[0], np.where(data_y == 1)[0])
                while len(positive_indices)==0:
                    fp_index = random_false_positive_index(labels, data_y)[0]
                    x_fp = data_x[:, fp_index]
                    positive_indices = np.intersect1d(np.where(S == S[fp_index])[0], np.where(data_y == 1)[0])
            else:
                return None, None, None, None, None
    w[-1] = -1 * np.dot(w[0:-1], x_fp[0:-1])
    positive_indices = np.intersect1d(np.where(S == S[fp_index])[0], np.where(data_y == 1)[0])
    epochs = 400
    learn_rate = 0.001
    count_k = 5
    k = count_k
    for i in range(0, epochs):
        start_print = 436
        start_obj = hyperplane_init_objective(w[0:-1], data_x[0:-1, positive_indices], data_y, x_fp[0:-1], positive_indices)
        indices = np.where(np.dot(w[0:-1], data_x[0:-1, positive_indices]) - np.dot(w[0:-1], x_fp[0:-1]) < 1)[0]
        gradient = np.sum(data_x[:, positive_indices][:, indices], axis=1) - len(indices) * x_fp.reshape(-1,)
        indices = np.where(np.dot(w[0:-1], data_x[0:-1, positive_indices]) - np.dot(w[0:-1], x_fp[0:-1]) < 0)[0]
        gradient += np.sum(data_x[:, positive_indices][:, indices], axis=1) - len(indices) * x_fp.reshape(-1,)
        w_new = np.zeros_like(w)
        w_new[0:-1] = w[0:-1] + learn_rate * gradient[0:-1]
        w_new[-1] = w[-1]
        end_obj = hyperplane_init_objective(w_new[0:-1], data_x[0:-1, positive_indices], data_y, x_fp[0:-1], positive_indices)
        if (i % 100 == 0) and (visualise == True):
            pdf.text(20, start_print, 'learn = ' + str(learn_rate))
            pdf.text(20, start_print + 16, 'Start objective value = ' + str(start_obj))
            pdf.text(20, start_print + 16 * 2, 'Hyperplane' + str(w))
            pdf.text(20, start_print + 16 * 3, 'derivative' + str(gradient))
            pdf.text(20, start_print + 16 * 4, 'End objective value = ' + str(end_obj))
            pdf.text(20, start_print + 16 * 5, 'false positive = ' + str(x_fp))
            pdf.text(20, start_print + 16 * 6, 'false positive set number = ' + str(S[fp_index]))
            start_print = start_print + 16 * 7
            sets = np.unique(S)
            for k in sets:
                n = np.intersect1d(np.where(data_y == 1)[0], np.where(S == k)[0])
                pdf.text(20, start_print, 'num_set_' + str(k) + '_points = ' + str(len(n)))
                start_print+=16
            if first:
                W_temp=[]
                W_temp.append(w.reshape(-1, 1))
                draw_init_lines(W_temp, S, w, data_x, data_y, i, pdf)
            else:
                draw_init_lines(W, S, w, data_x, data_y, i, pdf)
            pdf.add_page()
        if end_obj <= start_obj:
            k = count_k
            learn_rate = learn_rate / 2
        else:
            w = w_new
            k = k - 1
            if k <= 0:
                k = count_k
                learn_rate = learn_rate * 1.5
        # w[-1] = -1 * np.dot(w[0:-1], x_fp[0:-1])
    w[-1] = np.amax(-1 * np.dot(w[0:-1], data_x[0:-1, positive_indices]))
    #draw_init_lines(W, w, data_x, data_y, epochs, pdf)
    end_obj = hyperplane_init_objective(w[0:-1], data_x[0:-1, positive_indices], data_y, x_fp[0:-1], positive_indices)
    not_in_the_hull = (abs(end_obj - len(positive_indices)) <= 2)
    return w, not_in_the_hull, S[fp_index], x_fp, fp_index

def min_distance_point(point, data_x, data_y, label):
    label_indices = np.where(data_y==label)[0]
    dist = np.zeros(data_y) + np.Inf
    data_s = data_x[:, label_indices]
    dist[label_indices] = np.linalg.norm(data_s - point)
    min_index = np.argmin(dist)
    return min_index

def learn_hyperplanes(W, S, data_x, data_y, learn_rate, p, pdf, added_hyp, visualise = False):
    iter = 0
    #W_new = np.ones_like(W)
    #W = np.ones_like(W)
    epochs = 1000
    count_k = 10
    k=10
    count_zero_derivative = []
    count_zero_one_sided = []
    for i in range(0, len(W)):
            count_zero_derivative.append(np.zeros(W[i].shape[1]))
            count_zero_one_sided.append(np.zeros(W[i].shape[1]))

    #while not converged(W_new, W) and
    while iter<epochs:
        obj_start, dot_prods = l_theta(W, S, data_x, data_y, return_intermediates=1)
        derivative = gradient(W, S, data_x, data_y, dot_prods)
        if iter % 300 == 0 and visualise == True:
            draw_lines(W, S, data_x, data_y, p*epochs + iter, learn_rate, pdf, added_hyp, derivative)
            pdf.add_page()

        W_new = update_w(W, derivative, learn_rate)
        obj_end = l_theta(W_new, S, data_x, data_y)

        if(obj_end>obj_start):
            W = W_new
            k = k - 1
            if k <= 0:
                k = count_k
                learn_rate = learn_rate * 1.5
        else:
            k = count_k
            learn_rate = learn_rate / 2
        iter = iter + 1
    return W

def get_probs(W, data_x):
    predict_pos = []
    for i in range(0, len(W)):
        predict_pos.append(np.amin(np.dot(W[i].T, data_x), axis=0))
    return logistic(np.amax(np.asarray(predict_pos), axis=0))

def predict(W, data_x):
    predict_pos = []
    for i in range(0, len(W)):
        predict_pos.append(np.amin(np.dot(W[i].T, data_x), axis=0))

    predict_pos = np.amax(np.asarray(predict_pos), axis=0)
    prediction = np.where(predict_pos>0, 1, 0)
    #predict_neg = np.amax(np.dot(W.T, data_x), axis=0)
    #prediction = predict_pos + predict_neg
    return prediction

def draw_lines(W, S, data_x, data_y, iter, learn_rate, pdf, added_hyp, derivative):
    #dir_points = {0: [1.5, 0], 1: [1.5, 0.2], 2: [1.5, 0.4], 3: [1.5, 0.6], 4: [1.5, 0.8], 5: [1.5, 1.0]}

    # plt.scatter(data_x[:, 0], data_x[:, 1], c=colors)
    #color_map = {0: 'cyan', 1: 'orange', 2: 'green', 3: 'black', 4: 'magenta', 5: 'red', 6:'brown', 7:'lime', 8:'maroon'}
    labels = predict(W, data_x)
    true_positive_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 1))
    true_negative_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 0))
    false_positive_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 1))
    false_negative_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 0))

    plt.scatter(data_x[0, true_positive_indices], data_x[1, true_positive_indices], c='blue', s=1)
    plt.scatter(data_x[0, true_negative_indices], data_x[1, true_negative_indices], c='yellow', s=1)
    plt.scatter(data_x[0, false_positive_indices], data_x[1, false_positive_indices], c='lime', s=1)
    plt.scatter(data_x[0, false_negative_indices], data_x[1, false_negative_indices], c='pink', s=1)

    x_min, x_max = data_x[0, :].min() - 1, data_x[0, :].max() + 1
    y_min, y_max = data_x[1, :].min() - 1, data_x[1, :].max() + 1
    grain = 0.001
    x1 = np.arange(x_min, x_max, grain)
    obj = l_theta(W, S, data_x, data_y)
    pdf.text(20, 336, 'learn rate = ' + str(learn_rate))
    pdf.text(20, 352, 'Objective = ' + str(obj))
    k=0
    p=0
    for j in range(0, len(W)):
        for i in range(0, W[j].shape[1]):
            w = W[j][:, i]
            pdf.text(20, 352 + 14*(k+1), 'hyperplane_'+str(j) + '_' + str(w))
            pdf.text(20, 352 + 14*(k+2), 'hyperplane_derivative_'+str(j) + '_' + str(derivative[j][:, i]))
            k+=2
            if w[1] == 0:
                x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
            else:
                x2 = -1 * (w[2] + w[0] * x1) / w[1]
            if [j, i] in added_hyp:
                plt.plot(x1, x2, color='black', label='set ' + str(j))
            else:
                plt.plot(x1, x2, color=color_map[p], label = 'set '+ str(j))
            sgn = (w[0] * dir_points[p][0] + w[1] * dir_points[p][1] + w[2])
            if sgn > 0:
                plt.scatter(dir_points[p][0], dir_points[p][1], marker='+',
                            color=color_map[p])
            if sgn < 0:
                plt.scatter(dir_points[p][0], dir_points[p][1], marker='o',
                            color=color_map[p])
            plt.axis([x_min, x_max, y_min, y_max])
        p += 1
    plt.legend(loc='upper right')
    plt.savefig('hyp_'+ str(len(W)) + str(iter) + str(W[0][0,0]) + '.png')
    pdf.image('hyp_' + str(len(W)) + str(iter) + str(W[0][0,0]) + '.png', x=0, y=0, w=320,h=320)
    plt.gca().cla()

def draw_init_lines(W, S, w_init, data_x, data_y, iter, pdf):
    labels = predict(W, data_x)
    true_positive_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 1))
    true_negative_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 0))
    false_positive_indices = np.intersect1d(np.where(data_y == 0), np.where(labels == 1))
    false_negative_indices = np.intersect1d(np.where(data_y == 1), np.where(labels == 0))

    plt.scatter(data_x[0, true_positive_indices], data_x[1, true_positive_indices], c='blue', s=1)
    plt.scatter(data_x[0, true_negative_indices], data_x[1, true_negative_indices], c='yellow', s=1)
    plt.scatter(data_x[0, false_positive_indices], data_x[1, false_positive_indices], c='lime', s=1)
    plt.scatter(data_x[0, false_negative_indices], data_x[1, false_negative_indices], c='pink', s=1)

    x_min, x_max = data_x[0, :].min() - 1, data_x[0, :].max() + 1
    y_min, y_max = data_x[1, :].min() - 1, data_x[1, :].max() + 1
    grain = 0.001
    x1 = np.arange(x_min, x_max, grain)
    k = 0
    p = 0

    i = p+1
    w = w_init
    if w[1] == 0:
        x2 = -1 * (w[2] + w[0] * x1) / 0.000000000000000000001
    else:
        x2 = -1 * (w[2] + w[0] * x1) / w[1]
    plt.plot(x1, x2, color='black')
    sgn = (w[0] * dir_points[i][0] + w[1] * dir_points[i][1] + w[2])
    if sgn > 0:
        plt.scatter(dir_points[i][0], dir_points[i][1], marker='+',
                    color='black')
    if sgn < 0:
        plt.scatter(dir_points[i][0], dir_points[i][1], marker='o',
                    color='black')
    plt.axis([x_min, x_max, y_min, y_max])

    plt.savefig('hyp_init'+ str(w_init[0]) + str(iter) + '.png')
    pdf.image('hyp_init' + str(w_init[0]) + str(iter) + '.png', x=0, y=0, w=700,h=420)
    plt.gca().cla()

def draw_memberships(W, S, data_x, pdf):
    sets = np.unique(S)
    start_print = 436
    #color_map = {0: 'cyan', 1: 'orange', 2: 'green', 3: 'maroon', 4: 'magenta', 5: 'red', 6: 'brown', 7: 'lime', 8: 'maroon'}
    for set in sets:
        indices = np.where(S == set)[0]
        plt.scatter(data_x[0, indices], data_x[1, indices], c=color_map[set], s=1, label = 'set ' + str(set))
        pdf.text(20, start_print, 'Set ' + str(set) + ' size' + str(len(indices)))
        start_print+=16
    plt.legend(loc = 'upper right')
    plt.savefig('mem' + str(len(W)) + str(W[0][0, 0]) + '.png')
    pdf.image('mem' + str(len(W)) + str(W[0][0, 0]) + '.png', x=0, y=0, w=320, h=320)

def train_ucs(data_x, data_y, n_hyp, num_sep, out_file, visualise = False):
    W = []
    S = np.zeros(data_x.shape[1], dtype='int')

    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)

    first = True
    for i in range(0, n_hyp):
        W, added_hyperplanes, data_x, data_y = add_hyperplane(W, S, data_x, data_y, first, num_sep, pdf, visualise)
        if len(added_hyperplanes) == 0:
            return W
        for j in range(0, 10):
            # pdf.text(20, 436, 'E step')
            # pdf.add_page()
            S = refresh_memberships(W, data_x)
            # draw_memberships(W, S, data_x, pdf)
            # pdf.add_page()
            W = learn_hyperplanes(W, S, data_x, data_y, 0.001, 10 * i + j, pdf, added_hyperplanes, visualise)
            S = refresh_memberships(W, data_x)
            # draw_memberships(W, S, data_x, pdf)
            # pdf.add_page()
        first = False
    return W

def train_ucs_ablate_hyp_per_sep(data_x, data_y, n_hyp, num_sep, out_file, visualise = False):
    W = []
    S = np.zeros(data_x.shape[1], dtype='int')

    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)

    first = True
    for i in range(0, n_hyp):
        W, added_hyperplanes = add_hyperplane_1(W, S, data_x, data_y, first, num_sep, pdf, visualise)
        if len(added_hyperplanes) == 0:
            return W
        for j in range(0, 10):
            S = refresh_memberships(W, data_x)
            W = learn_hyperplanes(W, S, data_x, data_y, 0.001, 10 * i + j, pdf, added_hyperplanes, visualise)
            S = refresh_memberships(W, data_x)
        first = False
    return W

def train_existing(W, data_x, data_y, n_hyp, out_file, visualise = False):
    S = refresh_memberships(W, data_x)

    pdf = FPDF(orientation='L', unit='pt', format='A4')
    pdf.add_page()
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(0, 0, 0)

    first = False
    for i in range(0, n_hyp):
        W, added_hyperplanes = add_hyperplane(W, S, data_x, data_y, first, pdf, visualise)
        for j in range(0, 10):
            S = refresh_memberships(W, data_x)
            W = learn_hyperplanes(W, S, data_x, data_y, 0.001, 10 * i + j, pdf, added_hyperplanes, visualise)
            S = refresh_memberships(W, data_x)
    return W


def test(W, data_x):
    labels = predict(W, data_x)
    return labels


def main():
    for a in range(0, 1):
        dim = 3
        np.random.seed(a)
        data = pd.read_csv('../union_of_convex_datasets/checkerboard.csv', header=None).values
        data_x = np.hstack((data[:, 0:-1], np.ones(data.shape[0]).reshape(-1, 1))).T
        #data_x = half_moon_quadratic(data_x)
        data_y = data[:, -1]
        #data_y = np.where(data_y == -1, 1, -1)
        data_y = np.where(data_y == -1, 0, 1)
        print(np.sum(np.where(data_y == 1, 1, 0)))
        print(np.sum(np.where(data_y == 0, 1, 0)))
        current_dir = os.getcwd()
        if os.path.exists(os.path.join(current_dir, 'log')) is False:
            os.makedirs(os.path.join(current_dir, 'log'))
        os.chdir(os.path.join(current_dir, 'log'))

        W = []
        S = np.zeros(data_x.shape[1], dtype='int')
        epochs = 1


        pdf = FPDF(orientation='L', unit='pt', format='A4')
        pdf.add_page()
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(0, 0, 0)
        c = 2   #max number of convex sets
        first = True
        for i in range(0, 10):
            W, added_hyperplanes, data_x, data_y = add_hyperplane(W, S, data_x, data_y, first, c, pdf, visualise = False)
            for j in range(0, 10):
                pdf.text(20, 436, 'E step')
                pdf.add_page()
                S = refresh_memberships(W, data_x)
                draw_memberships(W, S, data_x, pdf)
                pdf.add_page()
                W = learn_hyperplanes(W, S, data_x, data_y, 0.001, 10*i+j, pdf, added_hyperplanes,  visualise = False)
                S = refresh_memberships(W, data_x)
                draw_memberships(W, S, data_x, pdf)
                pdf.add_page()
            first = False
        #pdf.output('results' + str(a) + '.pdf')
        for i in range(0, len(W)):
            print(W[i].shape)
        os.chdir(current_dir)

if __name__ == '__main__':
    main()
