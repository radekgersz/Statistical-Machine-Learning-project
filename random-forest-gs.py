import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from preprocessing import load_and_split


if __name__ == "__main__":
    print("Loading data...")
    path = "data/training_data_ht2025.csv"
    X_train, X_test, y_train, y_test = load_and_split(path)

    rf = RandomForestClassifier(random_state=23, class_weight='balanced')

    param_grid = {
        'n_estimators': [100, 200, 300],      
        'max_depth': [10, 15, 20, None],      
        'min_samples_split': [2, 5, 10],      
        'min_samples_leaf': [1, 2, 4],        
        'max_features': ['sqrt', 'log2']      
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=23) 

    grid_search = GridSearchCV(
        estimator = rf,
        param_grid = param_grid,
        cv = cv_strategy,
        n_jobs = -1,
        verbose = 1
    )

    print("Grid Search starting...\n")
    grid_search.fit(X_train, y_train) 

    print(f"\nBest Parameters found: {grid_search.best_params_}")
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")


    best_rf = grid_search.best_estimator_
    y_pred_optimized = best_rf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_optimized))
    print(classification_report(y_test, y_pred_optimized, target_names=['low_bike_demand', 'high_bike_demand']))

    y_pred_proba_optimized = best_rf.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_optimized)

    roc_auc = auc(fpr, tpr)
    print("ROC AUC:", roc_auc)

    # Plot
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("images/random_forest_roc_curve.png")
    plt.show()