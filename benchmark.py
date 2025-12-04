import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from preprocessing import load_and_split

def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)


    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    saveFig = False
    if saveFig:
        plt.savefig("images/benchmark_roc_curve.png")
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split("data/training_data_ht2025.csv")

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)

    print("Dummy Classifier Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["low_bike_demand", "high_bike_demand"], zero_division=0))

    # PLOT ROC CURVE
    plotROC = True
    if plotROC:
        y_pred_proba = dummy_clf.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_pred_proba)


