import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from preprocessing import get_dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import sys


def count_parameters(clf):
    coef = clf.coefs_
    inter = clf.intercepts_
    count = 0
    for i in range(0, len(coef)):
        count += coef[i].shape[0] * coef[i].shape[1]
        count += inter[i].shape[0]
    return count

def slp_run(data_name):
    output_dict = {}
    output_dict['dataset'] = data_name
    output_dict['algo'] = 'slp'
    data_x, data_y, data_x_test, data_y_test = get_dataset(data_name)
    output_dict['train_examples'] = data_x.shape[0]
    output_dict['columns'] = data_x.shape[1]
    output_dict['test_examples'] = data_x_test.shape[0]
    output_dict['positive_train_examples'] = np.sum(data_y)
    output_dict['positive_test_examples'] = np.sum(data_y_test)
    output_dict['negative_train_examples'] = len(data_y) - np.sum(data_y)
    output_dict['negative_test_examples'] = len(data_y_test) - np.sum(data_y_test)

    hidden_comb = []
    for i in range(5, 100, 5):
        hidden_comb.append((i,))
    #best {'hidden_layer_sizes': (10,)}
    parameters = {'hidden_layer_sizes': hidden_comb}
    nn = MLPClassifier(random_state=1, max_iter=10000)
    clf = GridSearchCV(nn, parameters, scoring='balanced_accuracy')
    clf.fit(data_x, data_y)
    output_dict['best_params'] = clf.best_params_
    clf = clf.best_estimator_
    scores = cross_val_score(clf, data_x, data_y, cv=5, scoring = 'balanced_accuracy')
    output_dict['mean'] = scores.mean()
    output_dict['std_dev'] = scores.std()
    clf.fit(data_x, data_y)

    predicted_labels = clf.predict(data_x)
    predicted_labels_test = clf.predict(data_x_test)
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
    roc_train = roc_auc_score(data_y, clf.predict_proba(data_x)[:, 1])
    roc_test = roc_auc_score(data_y_test, clf.predict_proba(data_x_test)[:, 1])
    output_dict['roc_train'] = roc_train
    output_dict['roc_test'] = roc_test
    print(output_dict)

slp_run(sys.argv[1])
