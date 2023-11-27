import numpy as np
from preprocessing import get_dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from ablation import test, train_ucs, get_probs, train_ucs_ablate_hyp_per_sep
from helpers import append_bias
import sys

np.random.seed(3)

def count_pararmeters(W):
    count=0
    count_hyp=0
    count_per_sep=[]
    for i in range(len(W)):
        count_hyp += W[i].shape[1]
        count_per_sep.append(W[i].shape[1])
        for j in range(W[i].shape[1]):
            count+=len(W[i][:, j])
    return count, count_hyp, count_per_sep

def ucs_run(data_name, c, f):
    output_dict = {}
    output_dict['dataset'] = data_name
    output_dict['algo'] = 'ucs'

    data_x, data_y, data_x_test, data_y_test = get_dataset(data_name)
    output_dict['train_examples'] = data_x.shape[0]
    output_dict['columns'] = data_x.shape[1]
    output_dict['test_examples'] = data_x_test.shape[0]
    output_dict['positive_train_examples'] = np.sum(data_y)
    output_dict['positive_test_examples'] = np.sum(data_y_test)
    output_dict['negative_train_examples'] = len(data_y) - np.sum(data_y)
    output_dict['negative_test_examples'] = len(data_y_test) - np.sum(data_y_test)
    output_dict['max_number_of_sets'] = c
    data_x_copy = np.copy(data_x)
    data_y_copy = np.copy(data_y)
    data_x_copy = data_x_copy.T
    data_x_test = data_x_test.T
    data_x_copy = append_bias(data_x_copy)
    data_x_test = append_bias(data_x_test)
    data_x = np.copy(data_x_copy)
    data_y = np.copy(data_y_copy)
    clf = train_ucs(data_x_copy, data_y_copy, 30, c, '--', visualise=False)
    c_n, c_h, c_p = count_pararmeters(clf)
    output_dict['number_of_hyperplanes'] = c_h
    output_dict['number_of_parameters'] = c_n
    output_dict['number_of_separators'] = len(clf)
    output_dict['hyperplanes_per_separator'] = c_p
    predicted_labels = test(clf, data_x)
    predicted_labels_test = test(clf, data_x_test)

    output_dict['train_accuracy'] = accuracy_score(data_y, predicted_labels)
    output_dict['train_balanced_accuracy'] = balanced_accuracy_score(data_y, predicted_labels)
    output_dict['train_recall'] = recall_score(data_y, predicted_labels)
    output_dict['train_precision'] = precision_score(data_y, predicted_labels)
    output_dict['train_f1_score'] = f1_score(data_y, predicted_labels)
    cf_train = confusion_matrix(data_y, predicted_labels)
    output_dict['train_true_positive'] = cf_train[1, 1]
    output_dict['train_false_positive'] = cf_train[0, 1]
    output_dict['train_false_negative'] = cf_train[1, 0]
    output_dict['train_true_negative'] = cf_train[0, 0]
    output_dict['test_accuracy'] = accuracy_score(data_y_test, predicted_labels_test)
    output_dict['test_balanced_accuracy'] = balanced_accuracy_score(data_y_test, predicted_labels_test)
    output_dict['test_recall'] = recall_score(data_y_test, predicted_labels_test)
    output_dict['test_precision'] = precision_score(data_y_test, predicted_labels_test)
    output_dict['test_f1_score'] = f1_score(data_y_test, predicted_labels_test)
    cf_test = confusion_matrix(data_y_test, predicted_labels_test)
    output_dict['test_true_positive'] = cf_test[1, 1]
    output_dict['test_false_positive'] = cf_test[0, 1]
    output_dict['test_false_negative'] = cf_test[1, 0]
    output_dict['test_true_negative'] = cf_test[0, 0]
    roc_train = roc_auc_score(data_y, get_probs(clf, data_x))
    roc_test = roc_auc_score(data_y_test, get_probs(clf, data_x_test))
    output_dict['roc_train'] = roc_train
    output_dict['roc_test'] = roc_test
    print(output_dict, file = f)


def run_ablation(data_name):
    for i in range(20, 25, 5):
        f = open("temp_" + "ablation_ucs_c=" +str(i) + " " + data_name + ".txt", "w")
        ucs_run(data_name, i, f)
        f.close()

#def run_ablation(data_name):
#    for i in range(3, 22, 3):
#        f = open("temp_" + "ablation_ucs_h=" +str(i) + " " + data_name + ".txt", "w")
#        ucs_run(data_name, i, f)
#        f.close()

#run_ablation("churn")
run_ablation(sys.argv[1])
