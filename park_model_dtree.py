import json
import time
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

training_times2 = []
for x in range(10):
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

    param_grid = {'max_depth': range(1, 31)}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid.fit(x_train, y_train)

    mean_test_scores = grid.cv_results_['mean_test_score']
    max_depths = param_grid['max_depth']

    plt.figure(figsize=(12, 6))
    plt.plot(max_depths, mean_test_scores, marker='o')
    plt.xlabel('Max Depth of Decision Tree')
    plt.ylabel('Mean Test Score (Cross-Validated Accuracy)')
    plt.title('Decision Tree Performance vs. Max Depth')
    plt.grid(True)
    plt.show()

    dump(grid.best_estimator_, 'decision_tree_model.joblib')

    accuracy = accuracy_score(y_test, grid.best_estimator_.predict(x_test))
    print(f"Accuracy of the best model: {accuracy * 100:.2f}%")

    y_pred = grid.predict(x_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['PD', 'NON_PD']))

    with open('decision_tree_accuracy.json', 'w') as f:
        json.dump({'accuracy': accuracy}, f)

    end_time = time.time()
    runtime = end_time - start_time
    training_times2.append(runtime)
    print(f"Runtime of the script: {runtime} seconds")

with open('dtree_training_times.json', 'w') as f:
    json.dump({'training_times': training_times2}, f)