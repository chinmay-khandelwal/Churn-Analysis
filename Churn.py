import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import metrics
from scipy.stats import boxcox
from scipy.stats import multivariate_normal

plt.style.use('bmh')
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.float_format', '{:,.2f}'.format)
np.set_printoptions(suppress=True)

# DATA UPLOAD
feature_list = ['rebate_ratio', 'monthly_revenue_ratio', 'mrr_type_A', 'mrr_type_B', 'chat_support_count', 'subscribers_A', 'subscribers_B']
full_feature_list = ['observation_date', 'churn_status'] + feature_list + ['addon_available']

data_path = r'/Users/yourname/Churn_simulated_data.xlsx'
simulated_churn_data = pd.read_excel(data_path, sheet_name='Sheet1')
simulated_churn_data = simulated_churn_data.astype({'churn_status': 'int',
                                                    'rebate_ratio': 'float',
                                                    'monthly_revenue_ratio': 'float',
                                                    'mrr_type_A': 'float', 'mrr_type_B': 'float',
                                                    'chat_support_count': 'float', 'subscribers_A': 'float',
                                                    'subscribers_B': 'float',
                                                    'addon_available': 'int'})
print(simulated_churn_data.head(-2))


# COHORT ANALYSIS

def cohort_plot(dataset, x_metric, y_metric, cohort_count=10, threshold=5):
    '''Cohorts are formed based on x_metric, and their means are used to visualize trends between x and y metrics.'''
    if len(dataset[x_metric].unique()) < threshold:
        cohort_x_means = list(dataset[x_metric].unique())
        cohort_y_means = dataset.groupby(x_metric)[y_metric].mean()
        data_to_plot = pd.DataFrame({x_metric: cohort_x_means, y_metric: cohort_y_means.values})
    else:
        groups = pd.qcut(dataset[x_metric], q=cohort_count, duplicates='drop')
        cohort_x_means = dataset.groupby(groups)[x_metric].mean()
        cohort_y_means = dataset.groupby(groups)[y_metric].mean()
        data_to_plot = pd.DataFrame({x_metric: cohort_x_means.values, y_metric: cohort_y_means.values})

    plt.figure(figsize=(6, 4))
    plt.plot(x_metric, y_metric, data=data_to_plot, marker='o', linewidth=2)
    plt.xlabel('Cohort Avg. of ' + x_metric)
    plt.ylabel('Cohort Avg. of ' + y_metric)
    plt.grid(visible=True)
    plt.title("Correlation: " + x_metric + " vs. " + y_metric)
    plt.show()


excluded_types = ['datetime64[ns]']  # Data types to exclude
valid_columns = []
for i in range(0, len(simulated_churn_data.dtypes), 1):
    if (str(simulated_churn_data.dtypes[i]) not in excluded_types) & (simulated_churn_data.columns[i] != 'churn_status'):
        valid_columns.append(simulated_churn_data.columns[i])

for metric in valid_columns:
    cohort_plot(simulated_churn_data, metric, 'churn_status', 10, 5)

# DATA SPLIT
simulated_churn_data['observation_date'] = pd.to_datetime(simulated_churn_data['observation_date'], format='%Y-%m-%d')
simulated_churn_data.sort_values(by='observation_date', ascending=True, inplace=True)
simulated_churn_data.reset_index(inplace=True, drop=True)

# Prepare features and target labels
if 'addon_available' not in feature_list:
    feature_list = feature_list + ['addon_available']
X = np.array(simulated_churn_data.loc[:, feature_list])
y = np.array(simulated_churn_data.loc[:, 'churn_status'])

# Time series split
time_split = TimeSeriesSplit(n_splits=3)

# LOGISTIC REGRESSION
# Hyperparameter tuning
log_model = LogisticRegression(solver='liblinear')
parameters = {'C': [.005, .01, .04, .08, .16, .32, .64, .75, .95],
              'penalty': ['l1', 'l2']}
log_search = GridSearchCV(log_model, param_grid=parameters, scoring='roc_auc', cv=time_split, verbose=1, n_jobs=-1)
log_search.fit(X, y)
log_results = pd.DataFrame(log_search.cv_results_)

print(log_search.best_score_)
print(log_search.best_params_)

# RANDOM FOREST
# Hyperparameter tuning
rf_model = RandomForestClassifier(n_jobs=-1)
rf_parameters = {'max_depth': [2, 5, 10],
                 'max_features': ['sqrt', 'log2'],
                 'n_estimators': [5, 10, 100, 500, 1000]}
rf_search = GridSearchCV(rf_model, param_grid=rf_parameters, scoring='roc_auc', cv=time_split, verbose=1, n_jobs=-1, refit=True)
rf_search.fit(X, y)
rf_results = pd.DataFrame(rf_search.cv_results_)

print(rf_search.best_score_)
print(rf_search.best_params_)

# Feature importance visualization
feature_names = feature_list
rf_features = pd.DataFrame(data=rf_search.best_estimator_.feature_importances_, index=feature_names, columns=['importance'])
rf_features.sort_values('importance').plot(kind='barh', title='Random Forest Feature Importance')
plt.show()

# XGBOOST
# Hyperparameter tuning
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1, eval_metric='auc')
xgb_parameters = {'max_depth': [2, 10, 20, 25, 40],
                  'learning_rate': [.1, .2, .5],
                  'n_estimators': [5, 10, 20, 30, 50],
                  'min_child_weight': [.05, .15, .5, 1]}
xgb_search = GridSearchCV(xgb_model, param_grid=xgb_parameters, scoring='roc_auc', cv=time_split, verbose=1, n_jobs=-1, refit=True)
xgb_search.fit(X, y)
xgb_results = pd.DataFrame(xgb_search.cv_results_)

print(xgb_search.best_score_)
print(xgb_search.best_params_)

# Feature importance visualization
rf_features = pd.DataFrame(data=xgb_search.best_estimator_.feature_importances_, index=feature_names, columns=['importance'])
rf_features.sort_values('importance').plot(kind='barh', title='XGBoost Feature Importance')
plt.show()