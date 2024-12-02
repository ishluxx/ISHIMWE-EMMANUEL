import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and explore data
data = pd.read_csv('HousingDataSet.csv')
print(data.head())
print(data.info())
print(data.describe())

# Visualize data
sns.pairplot(data)
plt.show()

# Preprocess data
data.fillna(method='ffill', inplace=True)
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop('AvPurchaseValue', axis=1)  # Replace 'target_column' with your actual target column
y = data['RequestFrequency']  # Replace 'target_column' with your actual target column

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Hyperparameter tuning with GridSearchCV
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Use best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate best model
y_pred = best_model.predict(X_test)
print('Accuracy of best model:', accuracy_score(y_test, y_pred))

# Cross-validation scores
scores = cross_val_score(best_model, X_train, y_train, cv=5)
print('Cross-Validation Scores:', scores)

