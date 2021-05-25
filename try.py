import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

if __name__ == '__main__':
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0.1, 0.4, 0.35, 0.8])
    p,r, _ = precision_recall_curve(labels, preds)
    auc = roc_auc_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('pr.png')

    plt.figure()
    plt.plot(fpr, tpr)
    # plt.xlabel('')
    # plt.ylabel('Precision')
    plt.title('ROC Curve')
    plt.savefig('roc.png')
