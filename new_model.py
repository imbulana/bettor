import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/games_cleaned.csv')

# Sort by date descending so latest games come first
df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
df.sort_values('GAME_DATE_EST', ascending=True, inplace=True)

# Compute per-team average stats for prediction via abbreviations
home_team_avgs = df.groupby('HOME_TEAM')[[
    'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home'
]].mean()
away_team_avgs = df.groupby('AWAY_TEAM')[[
    'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away'
]].mean()

# Features and target
features = [
    'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
    'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away'
]
X = df[features]
y = df['HOME_TEAM_WINS']

# Top 20% most recent games as test set
n_test = int(len(df) * 0.2)
X_test = X.iloc[:n_test]
y_test = y.iloc[:n_test]
X_train = X.iloc[n_test:]
y_train = y.iloc[n_test:]

# Define candidate models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    # 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    # 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    # 'SVC': SVC(probability=True, random_state=42)
}

# Evaluate each model via TimeSeriesSplit on training data
tscv = TimeSeriesSplit(n_splits=5)
results = {}
plt.figure(figsize=(10, 8))

for name, model in models.items():
    cv_aucs = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        pipeline.fit(X_tr, y_tr)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        cv_aucs.append(roc_auc_score(y_val, y_val_proba))

    avg_cv_auc = sum(cv_aucs) / len(cv_aucs)
    # Fit on full training set
    pipeline.fit(X_train, y_train)

    # Test evaluation
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={test_auc:.2f})')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

    # Classification report
    report = classification_report(y_test, y_test_pred)

    results[name] = {
        'pipeline': pipeline,
        'CV AUC': avg_cv_auc,
        'Test AUC': test_auc,
        'Report': report
    }

# Finalize ROC plot
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Test Set')
plt.legend()
plt.show()

# Print summary results
print("Model Evaluation Summary:")
for name, res in results.items():
    print(f"\n{name}:")
    print(f"  CV AUC: {res['CV AUC']:.4f}")
    print(f"  Test AUC: {res['Test AUC']:.4f}")
    print(f"  Classification Report:\n{res['Report']}")

# Select best model by Test AUC
best_name, best_res = max(results.items(), key=lambda kv: kv[1]['Test AUC'])
best_pipeline = best_res['pipeline']
print(f"\nBest model: {best_name} (AUC={best_res['Test AUC']:.2f})")

# Function to predict win prob from abbreviations
def predict_home_win_from_abbr(home_abbr, away_abbr, pipeline=best_pipeline):
    """Predicts probability that home team wins using team abbreviation averages."""
    if home_abbr not in home_team_avgs.index or away_abbr not in away_team_avgs.index:
        raise ValueError(f"Unknown team abbreviation: {home_abbr} or {away_abbr}")

    home_stats = home_team_avgs.loc[home_abbr]
    away_stats = away_team_avgs.loc[away_abbr]

    game_df = pd.DataFrame([{
        'FG_PCT_home': home_stats['FG_PCT_home'],
        'FT_PCT_home': home_stats['FT_PCT_home'],
        'FG3_PCT_home': home_stats['FG3_PCT_home'],
        'AST_home': home_stats['AST_home'],
        'REB_home': home_stats['REB_home'],
        'FG_PCT_away': away_stats['FG_PCT_away'],
        'FT_PCT_away': away_stats['FT_PCT_away'],
        'FG3_PCT_away': away_stats['FG3_PCT_away'],
        'AST_away': away_stats['AST_away'],
        'REB_away': away_stats['REB_away'],
    }])

    prob = pipeline.predict_proba(game_df)[0][1]
    print(f"Predicted win probability for {home_abbr} vs {away_abbr}: {prob:.2f}")
    return prob

# Example prediction
# predict_home_win_from_abbr('ORL', 'SAS')

# teams = [
#     'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls',
#     'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons',
#     'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 'Los Angeles Clippers',
#     'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks',
#     'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks',
#     'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns',
#     'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors',
#     'Utah Jazz', 'Washington Wizards'
# ]

teams = [
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI',
    'CLE', 'DAL', 'DEN', 'DET',
    'GSW', 'HOU', 'IND', 'LAC',
    'LAL', 'MEM', 'MIA', 'MIL',
    'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX',
    'POR', 'SAC', 'SAS', 'TOR',
    'UTA', 'WAS'
]

results = []

for home_team in teams:
    for away_team in teams:
        if home_team != away_team:
            predicted_probability = predict_home_win_from_abbr(home_team, away_team)
            results.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_win_prob': predicted_probability,
                'away_win_prob': 1 - predicted_probability
            })

# Convert to DataFrame
data = pd.DataFrame(results)

# Optionally format win_prob to 2 decimal places
data['home_win_prob'] = data['home_win_prob'].round(2)
data['away_win_prob'] = data['away_win_prob'].round(2)

# Save to CSV
data.to_csv('win_prob.csv', index=False)