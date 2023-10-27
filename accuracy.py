
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score
import numpy as np




def rocCurve(trueLabels, predictedProbs,numLayers, lookback, batch_size, hiddenLayerSize):
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(trueLabels[:, -1], predictedProbs[:, -1])

    # Calculate the AUC score
    roc_auc = roc_auc_score(trueLabels[:, -1], predictedProbs[:, -1])
    title ="ROCCurve_" + str(numLayers) + "_" + str(lookback)+ "_" + str(batch_size)+ "_" + str(hiddenLayerSize) 
    # Plot the ROC curve
    fig = plt.figure(title, figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    return fig

def precisionRecallCalc(trueLabels, predictedProbs):
    precision = precision_score(trueLabels[:, -1], predictedProbs)
    print(f"\nPrecision: {precision}")

    recall = recall_score(trueLabels[:, -1], predictedProbs)
    print(f"Recall (Sensitivity): {recall}\n")



def prCurve(trueLabels, predictedProbs,numLayers, lookback, batch_size, hiddenLayerSize):
    # Calculate the Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(trueLabels[:, -1], predictedProbs[:, -1])
    # Calculate the Area Under the Precision-Recall curve (AUC-PR)
    auc_pr = average_precision_score(trueLabels[:, -1], predictedProbs[:, -1])
    title ="PRCurve_" + str(numLayers) + "_" + str(lookback)+ "_" + str(batch_size)+ "_" + str(hiddenLayerSize) 
    # Plot the Precision-Recall curve
    fig = plt.figure(title, figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall Curve (AUC-PR = {auc_pr:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    return fig


def f1Score(trueLabels, predictLabels):
    # Calculate the F1-score
    f1 = f1_score(trueLabels[:, -1], predictLabels)
    print(f"\nF1-score: {f1}")
    

def graphF1Score(trueLabels, predictedProbs,numLayers, lookback, batch_size, hiddenLayerSize):
    # Create an array of different threshold values
    thresholds = np.arange(0.2, 1.0, 0.05)
    
    # Calculate F1 scores for each threshold value
    f1_scores = [f1_score(trueLabels[:,-1].astype(int), (predictedProbs[:,-1] >= threshold).astype(int)) for threshold in thresholds]
    title ="F1Score_" + str(numLayers) + "_" + str(lookback) + "_" + str(batch_size)+ "_" + str(hiddenLayerSize) 
    # Plot the F1 scores
    fig = plt.figure(title, figsize=(8, 6))
    plt.plot(thresholds, f1_scores, marker='o', linestyle='-')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    plt.grid(True)
    return fig
