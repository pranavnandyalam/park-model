import json
from scipy import stats

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
