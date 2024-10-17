from sklearn.metrics import precision_recall_fscore_support


def metric(pred, target):
    pre, rec, f1, sup = precision_recall_fscore_support(target, pred)
    f1 = sum(f1) / len(f1)
    return f1
