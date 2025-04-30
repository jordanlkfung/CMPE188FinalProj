import pandas as pd

all_cols = ['Unnamed: 0', 'id', 'photo', 'name', 'blurb', 'goal', 'pledged',
       'state', 'slug', 'disable_communication', 'country', 'currency',
       'currency_symbol', 'currency_trailing_code', 'deadline',
       'state_changed_at', 'created_at', 'launched_at', 'staff_pick',
       'backers_count', 'static_usd_rate', 'usd_pledged', 'creator',
       'location', 'category', 'profile', 'spotlight', 'urls', 'source_url',
       'friends', 'is_starred', 'is_backing', 'permissions', 'name_len',
       'name_len_clean', 'blurb_len', 'blurb_len_clean', 'deadline_weekday',
       'state_changed_at_weekday', 'created_at_weekday', 'launched_at_weekday',
       'deadline_month', 'deadline_day', 'deadline_yr', 'deadline_hr',
       'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr',
       'state_changed_at_hr', 'created_at_month', 'created_at_day',
       'created_at_yr', 'created_at_hr', 'launched_at_month',
       'launched_at_day', 'launched_at_yr', 'launched_at_hr',
       'create_to_launch', 'launch_to_deadline', 'launch_to_state_change']
cols_to_drop = ['Unnamed: 0','urls','name','blurb','id','slug',
                'photo','currency_trailing_code','friends','is_starred','is_backing','permissions','name_len','blurb_len','profile','static_usd_rate']
datetime_columns = ['state_changed_at','created_at', 'launched_at','deadline']

year_cols = ['created_at_yr', 'deadline_yr','state_changed_at_yr', 'launched_at_yr']

month_day_hr_cols = ['state_changed_at_month','created_at_day','created_at_month', 'created_at_hr', 'launched_at_month','launched_at_day','launched_at_hr',
        'deadline_month','deadline_day','deadline_hr', 'state_changed_at_day', 'state_changed_at_hr', ]
cat = ['deadline_weekday', 'created_at_weekday', 'state_changed_at_weekday', 'currency_symbol'
       'country', 'currency','source_url',
       'state','launched_at_weekday'
       ]

dtype_dict = {
    **{col: 'int16' for col in year_cols},
    **{col: 'int8' for col in month_day_hr_cols},
    **{col: 'category' for col in cat},
    'backers_count':'int32'
}

for i in datetime_columns:
    cols_to_drop.append(i)
cols_to_keep = [col for col in all_cols if col not in cols_to_drop]

df = pd.read_csv("kickstarter_data_with_features.csv", usecols=cols_to_keep, dtype=dtype_dict)

df['currency_symbol'] = df['currency_symbol'].astype('category').cat.codes
df['country'] = df['country'].astype('category').cat.codes
df['currency'] = df['currency'].astype('category').cat.codes
df['state'] = df['state'].astype('category').cat.codes
df['source_url'] = df['source_url'].astype('category').cat.codes #this probably has high correlation with category

df['launch_to_deadline'] = df['launch_to_deadline'].astype(str).str.extract(r'(\d+ days)')
df['create_to_launch'] = df['create_to_launch'].astype(str).str.extract(r'(\d+ days)')
df['launch_to_state_change'] = df['launch_to_state_change'].astype(str).str.extract(r'(\d+ days)')

df['name_len_clean'] = df['name_len_clean'].fillna(df['name_len_clean'].median())
df['name_len_clean'] = df['name_len_clean'].astype('int16')
df['blurb_len_clean'] = df['blurb_len_clean'].fillna(df['blurb_len_clean'].median())
df['blurb_len_clean'] = df['blurb_len_clean'].astype('int16')

df['launch_to_deadline'] = df['launch_to_deadline'].astype(str).str.extract(r'(\d+)')
df['create_to_launch'] = df['create_to_launch'].astype(str).str.extract(r'(\d+)')
df['launch_to_state_change'] = df['launch_to_state_change'].astype(str).str.extract(r'(\d+)')

df['launch_to_deadline'] = pd.to_numeric(df['launch_to_deadline'])
df['create_to_launch'] = pd.to_numeric(df['create_to_launch'])
df['launch_to_state_change'] = pd.to_numeric(df['launch_to_state_change'])

df['category'] = df['category'].astype('category').cat.codes

df['deadline_weekday'] = df['deadline_weekday'].cat.codes
df['state_changed_at_weekday'] = df['state_changed_at_weekday'].cat.codes
df['created_at_weekday'] = df['created_at_weekday'].cat.codes
df['launched_at_weekday'] = df['launched_at_weekday'].cat.codes

X = df.drop(columns=['state','creator','location','launch_to_state_change'], axis=1)
Y = df['state']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
numeric_cols = X.select_dtypes(include=['number']).columns
non_numeric_cols = X.select_dtypes(exclude=['number']).columns
scaler = StandardScaler()
X_scaled_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)
X_scaled = pd.concat([X_scaled_numeric, X[non_numeric_cols]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
param_grid = {'C': [0.1, 1, 10, 50, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'max_iter': [10000,25000, 50000, 75000, 100000 ],
              'kernel': ['rbf'],
              }
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
# grid.fit(X_train, y_train)
# print(grid.best_params_)

# print(grid.best_estimator_)

print("ADA")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_grid = {
    'n_estimators': [50, 100, 150, 200, 400, 600, 800, 1000],
    'learning_rate': [0.1,0.3, 0.5, 0.6, 0.8, 0.9],
    'base_estimator__max_depth': [1, 3, 4, 5, 6, 7],
    'base_estimator__min_samples_split': [2, 5, 7, 10]
}
ada =  AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
grid_search = GridSearchCV(ada, ada_grid, refit=True, verbose=3)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
print("Best parameters:", best_parameters)

# Get the best model
best_model = grid_search.best_estimator_
print("Best model:", best_model)