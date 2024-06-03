# Packages

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import f1_score, confusion_matrix
from tensor_data import prepare_image
from count_fingers_cnn import predict_result
import time

'''

Functions used to train model, check accuracy, visual results and features' correlation

'''

# Functions

def grid_search(features, labels, svm_params, lda_params, dt_params):
    """
    It searches for the best model and params for our datasets. Also, it uses cross validation to check on different sets.
    It saves the best models and give back the info
    :param features: pandas DataFrame    :param labels:
    :param svm_params: dictionary
    :param lda_params: dictionary
    :param dt_params: dictionary
    :return: print
    """

    svm_model = SVC()
    lda_model = LinearDiscriminantAnalysis()
    dt_model = DecisionTreeClassifier()

    models_params = [
        (svm_model, svm_params),
        (lda_model, lda_params),
        (dt_model, dt_params)
    ]

    for model, params in models_params:
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
        grid_search.fit(features, labels)

        estimator = grid_search.best_estimator_
        dump(estimator, f"{model}-model.joblib")

        print("Najlepsze parametry dla {}: {}".format(model.__class__.__name__, grid_search.best_params_))
        print("Najlepsza dokładność (accuracy) dla {}: {:.2f}".format(model.__class__.__name__, grid_search.best_score_))
        print()





def count_acc(data, model, filter):

    """
    It firsty preprocess data and then predicts results based on given pytorch model
    :param data: pandas DataFrame
    :param model: pytorch class nn.model
    :param filter: function
    :return: results, time, data
    """

    results = []

    test_tensor, time, data = prepare_image(data, filter)

    for X in test_tensor:
        X = X.unsqueeze(0)
        X = X.unsqueeze(0)

        result = predict_result(model, X)

        results.append(result)

    return results, time, data


def compare_time(data, model, filter):
    """
    That function counts time and repeats its 20 times
    :param data:
    :param model:
    :param filter:
    :return:
    """
    list_time = []
    time_sum = 0

    for _ in range(20):
        start_time = time.time()
        result, time_, data = count_acc(data, model, filter)
        end_time = time.time()

        time_result = end_time - start_time - time_

        list_time.append(time_result)
        time_sum += time_result

    return list_time, time_sum

def corr_visual(data):
    """
    It shows correlation matrix
    :param data: pandas Dataframe
    :return: plot

    """

    corr_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Macierz korelacji dla danych o palcach')
    plt.show()


def count_results(y_test, y_pred):
    """
    It shows results of model's work.
    F1_score and confusion matrix
    :param y_test: list    :param y_pred:
    :return: plot, print
    """

    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-score: {:.2f}".format(f1))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()