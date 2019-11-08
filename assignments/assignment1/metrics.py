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

    shape = prediction.shape[0]
    index_true_in_test = [index for index in range(shape) if ground_truth[index]]
    true_positive_index = [index for index in index_true_in_test if prediction[index]]
    false_negative_index = list(set(index_true_in_test) - set(true_positive_index))

    true_positive = len(true_positive_index)
    false_negative = len(false_negative_index)

    index_false_in_test = list(set(range(shape)) - set(index_true_in_test))
    false_positive_index = [index for index in index_false_in_test if not prediction[index]]
    true_negative_index = list(set(index_false_in_test) - set(false_positive_index))

    false_positive = len(false_positive_index)
    true_negative = len(true_negative_index)

    # Finally
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive + true_negative) \
               / (true_positive + true_negative + false_positive + false_negative)
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
