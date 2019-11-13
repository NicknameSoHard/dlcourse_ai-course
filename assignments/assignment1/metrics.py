def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    b = 1

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(ground_truth)):
        if prediction[i] == ground_truth[i] == 1:
            true_positive += 1
        if ground_truth[i] == 1 and prediction[i] != ground_truth[i]:
            false_positive += 1
        if prediction[i] == ground_truth[i] == 0:
            true_negative += 1
        if ground_truth[i] == 0 and prediction[i] != ground_truth[i]:
            false_negative += 1

    # Finally
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) / \
               (true_positive + true_negative + false_positive + false_negative)

    f1 = (1+b) * (precision * recall) / ((b**2 * precision) + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
