import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''


    true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

    for p, gt in zip(prediction, ground_truth):
        if p == gt == 1:
            true_positive += 1
        elif p == gt == 0:
            true_negative += 1
        elif p == 1 and gt == 0:
            false_positive += 1
        elif p == 0 and gt == 1:
            false_negative += 1

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    return np.mean([p == gt for p, gt in zip(prediction, ground_truth)])