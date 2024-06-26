import json
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

with open('svm_training_times.json', 'r') as f:
    svm_training_times = json.load(f)['training_times']

with open('dtree_training_times.json', 'r') as f:
    decision_tree_training_times = json.load(f)['training_times']

with open('lr_training_times.json', 'r') as f:
    logistic_regression_training_times = json.load(f)['training_times']

t_statistic_svm_dtree, p_value_svm_dtree = stats.ttest_rel(svm_training_times, decision_tree_training_times)
t_statistic_svm_lr, p_value_svm_lr = stats.ttest_rel(svm_training_times, logistic_regression_training_times)
t_statistic_dtree_lr, p_value_dtree_lr = stats.ttest_rel(decision_tree_training_times, logistic_regression_training_times)

print(f"SVM vs. Decision Tree t-statistic: {t_statistic_svm_dtree}, p-value: {p_value_svm_dtree}")
print(f"SVM vs. Logistic Regression t-statistic: {t_statistic_svm_lr}, p-value: {p_value_svm_lr}")
print(f"Decision Tree vs. Logistic Regression t-statistic: {t_statistic_dtree_lr}, p-value: {p_value_dtree_lr}")

algorithms = np.array(['Support Vector Machine', 'Decision Tree', 'Logistic Regression'])
times = np.array([99.313262438774, 123.23636357784, 22.404130291939,])
p_values = [p_value_svm_dtree, p_value_dtree_lr]

colors = ['skyblue', 'green', 'salmon']

plt.figure(figsize=(10, 6))
bar_width = 0.6

for i, p_value in enumerate(p_values):
    y_pos = max(times[i], times[i+1]) + max(times) * 0.05
    plt.plot([i, i, i+1, i+1], [y_pos, y_pos + 2, y_pos + 2, y_pos], lw=1.5, c='black')
    plt.text(i+0.5, y_pos + 5, f'p={p_value:.1e}', ha='center', va='bottom', color='black')

title_font = {'size':'16', 'color':'black', 'weight':'bold'}
axis_label_font = {'size':'14', 'color':'black', 'weight':'normal'}
ticks_font = {'size':'12', 'color':'black', 'weight':'normal'}

plt.title('Figure 2. Average Time (seconds) to Train Each Algorithm', **title_font)
barplot = plt.bar(algorithms, times, color=colors, width=bar_width)
plt.bar_label(barplot, labels=times, label_type='edge')
plt.ylabel('Time (seconds)', **axis_label_font)
plt.xlabel('Type of Classification Algorithm', **axis_label_font)

plt.ylim(0, max(times) * 1.15)
plt.tight_layout(pad=5)
plt.show()
