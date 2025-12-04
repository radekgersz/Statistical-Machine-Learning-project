# Benchmark

                   precision    recall  f1-score   support

 low_bike_demand       0.82      1.00      0.90       262
high_bike_demand       0.00      0.00      0.00        58

        accuracy                           0.82       320
       macro avg       0.41      0.50      0.45       320
    weighted avg       0.67      0.82      0.74       320


# Single tree
Decision Tree Accuracy: 0.83125

ROC AUC: 0.8433798367991576
Classification Report:
                   precision    recall  f1-score   support

 low_bike_demand       0.89      0.91      0.90       262
high_bike_demand       0.54      0.47      0.50        58

        accuracy                           0.83       320
       macro avg       0.71      0.69      0.70       320
    weighted avg       0.82      0.83      0.83       320


# Random forest with grid search

Best Parameters found: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 300}
Best F1 Score: 0.9031
Accuracy: 0.871875
ROC AUC: 0.8950381679389313

                  precision    recall  f1-score   support

 low_bike_demand       0.93      0.91      0.92       262
high_bike_demand       0.63      0.69      0.66        58

