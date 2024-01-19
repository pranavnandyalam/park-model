import json
import time
import os
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.io import imread
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler

# training_times3 = []
for x in range(1):
    start_time = time.time()
    Categories = ['PD', 'NON_PD']
    flat_data_arr = []
    target_arr = []
    datadir = '/Users/prn/PycharmProjects/park-model/pd_image_data'

    for i in Categories:
        print(f'loading... category : {i}')
        path = os.path.join(datadir, i)
        for img in os.listdir(path):
            if img.endswith('.png'):
                img_array = imread(os.path.join(path, img))
                img_resized = resize(img_array, (150, 150, 3))
                flat_data_arr.append(img_resized.flatten())
                target_arr.append(Categories.index(i))
        print(f'loaded category: {i} successfully')

    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)

    df = pd.DataFrame(flat_data)
    df['Target'] = target

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear']
    }

    lr = LogisticRegression(max_iter=5000)
    grid = GridSearchCV(lr, param_grid, cv=5, n_jobs=-1)
    grid.fit(x_train_scaled, y_train)

    dump(grid.best_estimator_, 'logistic_regression_model.joblib')
    best_index = grid.best_index_
    cv_results = grid.cv_results_
    n_folds = 5

    print(f"Cross-validation accuracies per fold for the best model:")
    for i in range(n_folds):
        fold_score = cv_results[f'split{i}_test_score'][best_index]
        print(f"Fold {i + 1}: {fold_score * 100:.2f}%")

    y_pred = grid.best_estimator_.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, grid.best_estimator_.predict(x_test))
    print(f"The model is {accuracy * 100}% accurate")
    report = classification_report(y_test, y_pred, target_names=Categories)
    print(report)

    with open('logistic_regression_accuracy.json', 'w') as f:
        json.dump({'accuracy': accuracy}, f)

    y_pred = grid.predict(x_test)
    score = accuracy_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}%'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()

    y_prob = grid.best_estimator_.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    end_time = time.time()
    runtime = end_time - start_time
#     training_times3.append(runtime)
#     print(f"Runtime of the script: {runtime} seconds")
# with open('lr_training_times.json', 'w') as f:
#     json.dump({'training_times': training_times3}, f)