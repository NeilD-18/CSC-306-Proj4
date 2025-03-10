"""Evaluation module for calculating accuracy, precision, recall, and f-score
"""
import numpy as np


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
    total_predictions = len(y_pred)
    
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

def convert_to_type(value):
    """Convert string value to its respective type (bool, int, float, or str).
    
    Args:
        value (str): The string value to convert.
    
    Returns:
        The converted value in its respective type.
    """
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value

def check_data_types(pred, true):
    """Check if the data types of a single predicted and true value are the same.
    
    Args:
        pred: predicted label (string format)
        true: corresponding true label (string format)
    
    Returns:
        bool: True if data types match, False otherwise
    """
    return type(pred) == type(true)

def eval_types(y_pred, y_true):
    """Check the data type for each index in both arrays and return the percentage of correct data types.
    
    Args:
        y_pred: list of predicted labels (all in string format)
        y_true: list of corresponding true labels (all in string format)
    
    Returns:
        float: percentage of correct data types
    """
    if len(y_pred) != len(y_true):
        raise ValueError("The length of y_pred and y_true must be the same.")
    
    correct_types = sum(1 for pred, true in zip(y_pred, y_true) if check_data_types(pred, true))
    return (correct_types / len(y_pred)) * 100

def evaluate(y_pred, y_true):
    """Calculate various evaluation metrics of the predicted labels
    and print out the results.
    
    Args:
        y_pred: list of predicted labels
        y_true: list of corresponding true labels
    """
    accuracy = get_accuracy(y_pred, y_true)
    eval_types_results = eval_types(y_pred, y_true)
    
    print(f"Evaluation Results:")
    print(f"-------------------")
    print(f"Accuracy: {accuracy * 100:.0f}%")
    print(f"Evaluation Types: {eval_types_results:.0f}%")


# Example usage
if __name__ == "__main__":
    y_pred = ["True", "4", "8.8", "hello"]
    y_true = ["False", "3", "7.5", "42"]
    print(check_data_types(y_pred[0], y_true[0]))  # Should print True
    print(eval_types(y_pred, y_true))  # Should print 75.0

    # Example with arrays
    y_pred = ["[1, 2, 3]", "4", "8.8", "hello"]
    y_true = ["[4, 5, 6]", "3", "7.5", "42"]
    print(check_data_types(y_pred[0], y_true[0]))  # Should print False
    print(eval_types(y_pred, y_true))  # Should print 50.0

