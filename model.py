import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("nba_games_dataset.csv")

# Define target and features
target = 'Home_PTS'
X = df.drop(columns=['Month_Day', 'Home_Team', 'Away_Team', target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
'Linear Regression': LinearRegression(),
'Ridge Regression': Ridge(alpha=1.0),
'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train, predict, and evaluate
results = []
for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        results.append({'Model': name, 'MSE': mse, 'R2 Score': r2})

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Linear Regression is the best model