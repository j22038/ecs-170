import numpy as np

# Function for computing F1_Score
# Parameters: y_actual - truth labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 label, test image 2 label, test image 3 label, ...]
#             y_predicted - predicted labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 predicted label, test image 2 predicted label, test image 3 predicted label, ...]
# Returns: F1_Score - F1_Score for each class
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

# Function for computing confusion matrix
# Parameters: y_actual - truth labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 label, test image 2 label, test image 3 label, ...]
#             y_predicted - predicted labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 predicted label, test image 2 predicted label, test image 3 predicted label, ...]
# Returns: confusion_matrix [4x4] - confusion matrix for each class
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


# Function for computing precision
# Parameters: y_actual - truth labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 label, test image 2 label, test image 3 label, ...]
#             y_predicted - predicted labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 predicted label, test image 2 predicted label, test image 3 predicted label, ...]
# Returns: precision_scores [4]
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


# Function for computing recall
# Parameters: y_actual - truth labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 label, test image 2 label, test image 3 label, ...]
#             y_predicted - predicted labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 predicted label, test image 2 predicted label, test image 3 predicted label, ...]
# Returns: recall_scores [4]
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


# Function for computing accuracy
# Parameters: y_actual - truth labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 label, test image 2 label, test image 3 label, ...]
#             y_predicted - predicted labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 predicted label, test image 2 predicted label, test image 3 predicted label, ...]
# Returns: accuracy
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


# Function for computing macro F1
# Parameters: y_actual - truth labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 label, test image 2 label, test image 3 label, ...]
#             y_predicted - predicted labels(ex, 0 = Healthy, 1 = Disease 1, 2 = Disease 2, 3 = Disease 3) [test image 1 predicted label, test image 2 predicted label, test image 3 predicted label, ...]
# Returns: macro_f1 - averaged F1 score across all classes
def macro_F1(y_actual, y_predicted):

    f1_scores = F1_Score(y_actual, y_predicted)
    
    macro_f1 = np.mean(f1_scores)
    
    return macro_f1

