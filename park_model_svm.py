import json
import time
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from joblib import dump

# training_times = []
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
    df.shape

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,
                                                        random_state=77, stratify=y)
    print("Indices of samples in the test set:")
    print(x_test.index.tolist())

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }

    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)

    model.fit(x_train, y_train)

    model.fit(x_train, y_train)
    dump(model.best_estimator_, 'svm_model.joblib')

    best_index = model.best_index_
    cv_results = model.cv_results_
    n_folds = 5

    print(f"Cross-validation accuracies per fold for the best model:")
    for i in range(n_folds):
        fold_score = cv_results[f'split{i}_test_score'][best_index]
        print(f"Fold {i + 1}: {fold_score * 100:.2f}%")

    results = {'test_indices': x_test.index.tolist(), 'predictions': model.predict(x_test).tolist()}
    with open('svm_results.json', 'w') as f:
        json.dump(results, f)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    with open('svm_accuracy.json', 'w') as f:
        json.dump({'accuracy': accuracy}, f)
    with open('categories.txt', 'w') as f:
        for category in Categories:
            f.write(f'{category}\n')

    results = model.cv_results_
    results_df = pd.DataFrame(results)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(results_df)), results_df['mean_test_score'], marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Test Score')
    plt.title('Results')
    plt.grid(True)
    plt.show()

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"The model is {accuracy * 100}% accurate")

    print(classification_report(y_test, y_pred, target_names=['PD', 'NON_PD']))

    end_time = time.time()
    runtime = end_time - start_time
    # training_times.append(runtime)
    # print(f"Runtime of the script: {runtime} seconds")

# with open('svm_training_times.json', 'w') as f:
#     json.dump({'training_times': training_times}, f)