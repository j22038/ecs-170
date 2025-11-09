import numpy as np

def F1_Score(y_actual, y_predicted):

    precision_scores = precision(y_actual, y_predicted)
    recall_scores = recall(y_actual, y_predicted)
    
    f1_scores = np.zeros(4, dtype=float)
    
    for i in range(4):
        if precision_scores[i] + recall_scores[i] == 0:
            f1_scores[i] = 0.0
        else:
            f1_scores[i] = 2 * (precision_scores[i] * recall_scores[i]) / (precision_scores[i] + recall_scores[i])
    
    return f1_scores


def confusion_matrix(y_actual, y_predicted):
    
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)
    
    cm = np.zeros((4, 4), dtype=int)

    for i in range(len(y_actual)):
        actual_label = y_actual[i]
        predicted_label = y_predicted[i]

        if 0 <= actual_label <= 3 and 0 <= predicted_label <= 3:
            cm[actual_label, predicted_label] += 1
    
    return cm


def precision(y_actual, y_predicted):

    cm = confusion_matrix(y_actual, y_predicted)
    
    precision_scores = np.zeros(4, dtype=float)
    
    for i in range(4):

        column_sum = np.sum(cm[:, i])
        
        if column_sum == 0:
            precision_scores[i] = 0.0
        else:
            precision_scores[i] = cm[i, i] / column_sum
    
    return precision_scores


def recall(y_actual, y_predicted):

    cm = confusion_matrix(y_actual, y_predicted)
    
    recall_scores = np.zeros(4, dtype=float)
    
    for i in range(4):

        row_sum = np.sum(cm[i, :])
        
        if row_sum == 0:
            recall_scores[i] = 0.0
        else:
            recall_scores[i] = cm[i, i] / row_sum
    
    return recall_scores


def accuracy(y_actual, y_predicted):

    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)
    
    if len(y_actual) == 0:
        return 0.0
    
    correct = 0
    total = len(y_actual)
    
    for i in range(total):
        if y_actual[i] == y_predicted[i]:
            correct += 1
    
    return correct / total


def macro_F1(y_actual, y_predicted):

    f1_scores = F1_Score(y_actual, y_predicted)
    
    macro_f1 = np.mean(f1_scores)
    
    return macro_f1

