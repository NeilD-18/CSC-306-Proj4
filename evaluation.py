"""Evaluation module for calculating accuracy, precision, recall, and f-score
"""
def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    correct = sum(1 for i in range(len(y_pred)) if y_pred[i] == y_true[i])
    accuracy = correct / len(y_pred)
    return accuracy

def get_precision(y_pred, y_true):
    """Calculate the precision for any type of labels.
    
    Args:
        y_pred: list of predicted labels (can be any type)
        y_true: list of corresponding true labels
    
    Returns:
        float: precision score
    """
    # True positives are where prediction matches truth
    true_positives = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
    print(f"True Positives: {true_positives}")
    # Total predictions is just length of predictions
    total_predictions = len(y_pred)
    print(f"Total Predictions: {total_predictions}")
    
    return true_positives / total_predictions 

def get_recall(y_pred, y_true):
    """Calculate the recall for any type of labels.
    
    Args:
        y_pred: list of predicted labels (can be any type)
        y_true: list of corresponding true labels
    
    Returns:
        float: recall score
    """
    # For non-binary classification, recall is same as precision
    # as we're just measuring correct predictions against total actual cases
    return get_precision(y_pred, y_true)

def get_fscore(y_pred, y_true):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return fscore

def evaluate(y_pred, y_true):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    accuracy = get_accuracy(y_pred, y_true)
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    
    print(f"Evaluation Results:")
    print(f"-------------------")
    print(f"Accuracy: {accuracy * 100:.0f}%")
    print(f"Precision: {precision * 100:.0f}%")
    print(f"Recall: {recall * 100:.0f}%")
    print(f"F-score: {fscore * 100:.0f}%")
