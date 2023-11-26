import numpy as np
from sklearn.model_selection import cross_val_score
from preprocessing import get_dataset
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import sys


def xg_run(data_name):
    output_dict = {}
    output_dict['dataset'] = data_name
    output_dict['algo'] = 'xg'
    data_x, data_y, data_x_test, data_y_test = get_dataset(data_name)
    output_dict['train_examples'] = data_x.shape[0]
    output_dict['columns'] = data_x.shape[1]
    output_dict['test_examples'] = data_x_test.shape[0]
    output_dict['positive_train_examples'] = np.sum(data_y)
    output_dict['positive_test_examples'] = np.sum(data_y_test)
    output_dict['negative_train_examples'] = len(data_y) - np.sum(data_y)
    output_dict['negative_test_examples'] = len(data_y_test) - np.sum(data_y_test)
    #parameters = {'n_estimators' : [20], 'max_depth' : [20], 'subsample' : [0.4]} #best
    #parameters = {'n_estimators' : [12, 14, 15, 16, 17], 'max_depth' : [6, 8, 10, 12, 14, 16], 'subsample' : [0.4, 0.5, 0.6, 0.7, 0.8]}
    #parameters = {'n_estimators' : [20, 22, 24, 26, 28, 30], 'max_depth' : [20, 22, 24, 26, 28, 30], 'subsample' : [0.2, 0.3, 0.4, 0.5]}
    #parameters = {'n_estimators' : [10, 12, 14, 16, 18, 20, 22], 'max_depth' : [20, 22, 24, 26, 28, 30], 'subsample' : [0.2, 0.3, 0.4, 0.5]}
    parameters = {'n_estimators' : [20, 22, 24, 26, 28, 30], 'max_depth' : [10, 12, 14, 16, 18, 20], 'subsample' : [0.2, 0.3, 0.4, 0.5]}
    random_forest = XGBClassifier(random_state=1, eval_metric='logloss', use_label_encoder=False)
    clf = GridSearchCV(random_forest, parameters, scoring='balanced_accuracy')
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

xg_run(sys.argv[1])