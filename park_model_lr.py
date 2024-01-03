import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

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

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
accuracies = []


for C in C_values:
    lr_classifier = LogisticRegression(C=C, max_iter=1000)
    lr_classifier.fit(x_train, y_train)
    y_pred = lr_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f"Classification Report for C={C}:")
    print(classification_report(y_test, y_pred, target_names=['PD', 'NON_PD']))

dump(lr_classifier, 'logistic_regression_model.joblib')

plt.figure(figsize=(12, 6))
plt.plot(C_values, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('Regularization Strength (C)')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Accuracy vs. Regularization Strength')
plt.grid(True)
plt.show()
