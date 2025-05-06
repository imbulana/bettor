import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def load_and_prepare_data(csv_file='total_data.csv'):
    """
    Loads and preprocesses the NBA game data.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame, or None on error.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file}")
        return None

    # Drop rows with missing values
    df.dropna(inplace=True)

    return df

def create_model_and_pipeline(df, model='logistic_regression'):
    """
    Creates a model and preprocessing pipeline.

    Args:
        df (pd.DataFrame): The input DataFrame.
        model (str, optional): The type of model to use.
            Options are: 'logistic_regression', 'random_forest', 'gradient_boosting', 'svm', 'naive_bayes', 'decision_tree'
            Defaults to 'logistic_regression'.

    Returns:
        tuple: (Pipeline, list) - The pipeline and list of numerical features, or (None, None) on error.
    """
    if df is None:
        return None, None

    # Define features
    numerical_features = [
        'Home_FG', 'Home_FGA', 'Home_FG%', 'Home_3P', 'Home_3PA', 'Home_3P%', 'Home_2P', 'Home_2PA', 'Home_2P%',
        'Home_FT', 'Home_FTA', 'Home_FT%', 'Home_ORB', 'Home_DRB', 'Home_TRB', 'Home_AST', 'Home_STL', 'Home_BLK',
        'Home_TOV', 'Home_PTS', 'Home_Season_W', 'Home_Season_L', 'Home_Season_Win_Pct', 'Home_Season_PTS_PG',
        'Home_Season_Opp_PTS_PG', 'Home_ORtg', 'Home_DRtg', 'Home_NRtg', 'Home_Pace', 'Home_Eff_FG%',
        'Home_True_Shooting_Pct',
        'Away_Opp_FG', 'Away_Opp_FGA', 'Away_Opp_FG%', 'Away_Opp_3P', 'Away_Opp_3PA', 'Away_Opp_3P%',
        'Away_Opp_2P', 'Away_Opp_2PA', 'Away_Opp_2P%', 'Away_Opp_FT', 'Away_Opp_FTA', 'Away_Opp_FT%',
        'Away_Opp_ORB', 'Away_Opp_DRB', 'Away_Opp_TRB', 'Away_Opp_AST', 'Away_Opp_STL', 'Away_Opp_BLK',
        'Away_Opp_TOV', 'Away_Opp_PTS', 'Away_Season_W', 'Away_Season_L', 'Away_Season_Win_Pct',
        'Away_Season_PTS_PG', 'Away_Season_Opp_PTS_PG', 'Away_ORtg', 'Away_DRtg', 'Away_NRtg', 'Away_Pace',
        'Away_Eff_FG%', 'Away_True_Shooting_Pct',
        'Home_Season_Point_Diff', 'Away_Season_Point_Diff', 'Home_Offensive_Advantage',
        'Away_Defensive_Advantage', 'Pace_Diff'
    ]
    categorical_features = ['Home_Team', 'Away_Team']

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Model selection
    if model == 'logistic_regression':
        classifier = LogisticRegression(solver='liblinear', random_state=42)
    elif model == 'random_forest':
        classifier = RandomForestClassifier(random_state=42)
    elif model == 'gradient_boosting':
        classifier = GradientBoostingClassifier(random_state=42)
    elif model == 'svm':
        classifier = SVC(probability=True, random_state=42)  # Enable probability estimates for ROC AUC
    elif model == 'naive_bayes':
        classifier = GaussianNB()
    elif model == 'decision_tree':
        classifier = DecisionTreeClassifier(random_state=42)
    else:
        print(f"Error: Model '{model}' not recognized. Using Logistic Regression.")
        classifier = LogisticRegression(solver='liblinear', random_state=42)

    # Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline, numerical_features

def train_and_evaluate_model(pipeline, df, model_name, numerical_features):
    """
    Trains and evaluates the  model.

    Args:
        pipeline (Pipeline): The pre-built pipeline.
                df (pd.DataFrame): The input DataFrame.
                model_name (str): Name of the model
                numerical_features (list):  List of numerical features.

    Returns:
        tuple: (model, report, auc_roc, avg_accuracy, y_test, y_pred, losses)
                Returns trained model, classification report, AUC-ROC score,
                cross-validation accuracy, and y_test, y_pred, and losses
    """
    if pipeline is None or df is None:
        return None, None, None, None, None, None, None

    X = df.drop('Home_Team_Wins', axis=1)
    y = df['Home_Team_Wins']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict probabilities
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of Home Team Wins
    y_pred = pipeline.predict(X_test)

    # Evaluate
    report = classification_report(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring='accuracy')  # 10-fold CV
    avg_accuracy = cv_scores.mean()

    print(f"Model: {model_name}")
    print("Classification Report:\n", report)
    print("AUC-ROC Score:", auc_roc)
    print("Average Cross-Validation Accuracy:", avg_accuracy)

    return pipeline, report, auc_roc, avg_accuracy, y_test, y_pred, [], X_test  # Return empty losses


def plot_roc_curve(y_test, y_pred_proba, model_name='Logistic Regression'):
    """
    Plots the ROC curve.

    Args:
        y_test (pd.Series): The true target values.
        y_pred_proba (np.array): Predicted probabilities.
        model_name (str, optional): Name of the model for the plot legend.  Defaults to 'Logistic Regression'.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_test, y_pred, model_name='Logistic Regression'):
    """
    Plots the confusion matrix.

    Args:
        y_test (pd.Series): The true target values.
        y_pred (np.array): Predicted target values.
        model_name (str, optional): Name of the model for the plot title. Defaults to 'Logistic Regression'.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

def plot_loss_curve(losses, model_name='Logistic Regression'):
    """
    Plots the loss curve.

    Args:
        losses (list): List of loss values during training.
        model_name (str, optional): Name of the model for the plot title. Defaults to 'Logistic Regression'.
    """
    if losses:
        plt.figure(figsize=(8, 6))
        plt.plot(losses, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve - {model_name}')
        plt.legend()
        plt.show()
    else:
        print(f"No loss data to plot for {model_name}.")

def predict_winner(pipeline, df, home_team, away_team, predict_prob='Home'):
    """
    Predicts the winner of a single game, given home and away teams.

    Args:
        pipeline (Pipeline): Trained pipeline
        df (pd.DataFrame):  The input DataFrame.
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.
        predict_prob (str, optional): Which probability to return.
            'Home' for probability of home team winning,
            'Away' for probability of away team winning.
            Defaults to 'Home'.

    Returns:
        float: Probability of the specified team winning, or None on error.
    """
    if pipeline is None:
        print("Error: Model pipeline is not trained.")
        return None

    # Get team stats.  Use .iloc[0] to get the first row as a dictionary.
    try:
        home_team_stats = df[df['Home_Team'] == home_team].iloc[0].to_dict()
        away_team_stats = df[df['Away_Team'] == away_team].iloc[0].to_dict()
    except IndexError:
        print(f"Error: Could not find team stats for {home_team} or {away_team}.")
        return None

    # Construct a game data dictionary in the format the model expects
    try:
        game_data = {
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Home_FG': home_team_stats['Home_FG'], 'Home_FGA': home_team_stats['Home_FGA'], 'Home_FG%': home_team_stats['Home_FG%'],
            'Home_3P': home_team_stats['Home_3P'], 'Home_3PA': home_team_stats['Home_3PA'], 'Home_3P%': home_team_stats['Home_3P%'],
            'Home_2P': home_team_stats['Home_2P'], 'Home_2PA': home_team_stats['Home_2PA'], 'Home_2P%': home_team_stats['Home_2P%'],
            'Home_FT': home_team_stats['Home_FT'], 'Home_FTA': home_team_stats['Home_FTA'], 'Home_FT%': home_team_stats['Home_FT%'],
            'Home_ORB': home_team_stats['Home_ORB'], 'Home_DRB': home_team_stats['Home_DRB'], 'Home_TRB': home_team_stats['Home_TRB'],
            'Home_AST': home_team_stats['Home_AST'], 'Home_STL': home_team_stats['Home_STL'], 'Home_BLK': home_team_stats['Home_BLK'],
            'Home_TOV': home_team_stats['Home_TOV'], 'Home_PTS': home_team_stats['Home_PTS'],
            'Home_Season_W': home_team_stats['Home_Season_W'], 'Home_Season_L': home_team_stats['Home_Season_L'],
            'Home_Season_Win_Pct': home_team_stats['Home_Season_Win_Pct'],
            'Home_Season_PTS_PG': home_team_stats['Home_Season_PTS_PG'],
            'Home_Season_Opp_PTS_PG': home_team_stats['Home_Season_Opp_PTS_PG'], 'Home_ORtg': home_team_stats['Home_ORtg'],
            'Home_DRtg': home_team_stats['Home_DRtg'], 'Home_NRtg': home_team_stats['Home_NRtg'], 'Home_Pace': home_team_stats['Home_Pace'],
            'Home_Eff_FG%': home_team_stats['Home_Eff_FG%'], 'Home_True_Shooting_Pct': home_team_stats['Home_True_Shooting_Pct'],
            'Away_Opp_FG': away_team_stats['Away_Opp_FG'], 'Away_Opp_FGA': away_team_stats['Away_Opp_FGA'],
            'Away_Opp_FG%': away_team_stats['Away_Opp_FG%'], 'Away_Opp_3P': away_team_stats['Away_Opp_3P'],
            'Away_Opp_3PA': away_team_stats['Away_Opp_3PA'], 'Away_Opp_3P%': away_team_stats['Away_Opp_3P%'],
            'Away_Opp_2P': away_team_stats['Away_Opp_2P'], 'Away_Opp_2PA': away_team_stats['Away_Opp_2PA'],
            'Away_Opp_2P%': away_team_stats['Away_Opp_2P%'], 'Away_Opp_FT': away_team_stats['Away_Opp_FT'],
            'Away_Opp_FTA': away_team_stats['Away_Opp_FTA'], 'Away_Opp_FT%': away_team_stats['Away_Opp_FT%'],
            'Away_Opp_ORB': away_team_stats['Away_Opp_ORB'], 'Away_Opp_DRB': away_team_stats['Away_Opp_DRB'],
            'Away_Opp_TRB': away_team_stats['Away_Opp_TRB'], 'Away_Opp_AST': away_team_stats['Away_Opp_AST'],
            'Away_Opp_STL': away_team_stats['Away_Opp_STL'], 'Away_Opp_BLK': away_team_stats['Away_Opp_BLK'],
            'Away_Opp_TOV': away_team_stats['Away_Opp_TOV'], 'Away_Opp_PTS': away_team_stats['Away_Opp_PTS'],
            'Away_Season_W': away_team_stats['Away_Season_W'], 'Away_Season_L': away_team_stats['Away_Season_L'],
            'Away_Season_Win_Pct': away_team_stats['Away_Season_Win_Pct'],
            'Away_Season_PTS_PG': away_team_stats['Away_Season_PTS_PG'],
            'Away_Season_Opp_PTS_PG': away_team_stats['Away_Season_Opp_PTS_PG'], 'Away_ORtg': away_team_stats['Away_ORtg'],
            'Away_DRtg': away_team_stats['Away_DRtg'], 'Away_NRtg': away_team_stats['Away_NRtg'], 'Away_Pace': away_team_stats['Away_Pace'],
            'Away_Eff_FG%': away_team_stats['Away_Eff_FG%'], 'Away_True_Shooting_Pct': away_team_stats['Away_True_Shooting_Pct'],
            'Home_Season_Point_Diff': home_team_stats['Home_Season_PTS_PG'] - home_team_stats['Home_Season_Opp_PTS_PG'],
            'Away_Season_Point_Diff': away_team_stats['Away_Season_PTS_PG'] - away_team_stats['Away_Season_Opp_PTS_PG'],
            'Home_Offensive_Advantage': home_team_stats['Home_ORtg'] - away_team_stats['Away_DRtg'],
            'Away_Defensive_Advantage': away_team_stats['Away_DRtg'] - home_team_stats['Home_ORtg'],
            'Pace_Diff': home_team_stats['Home_Pace'] - away_team_stats['Away_Pace']
        }
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Please check the column names in your data file.")
        return None

    # Convert game_data to dataframe
    game_df = pd.DataFrame([game_data])

    # Make prediction
    probability = pipeline.predict_proba(game_df)[0][1]  # Probability of home team win

    if predict_prob == 'Home':
        return probability
    elif predict_prob == 'Away':
        return 1 - probability
    else:
        print("Error: predict_prob must be 'Home' or 'Away'.")
        return None



def get_past_results(df, home_team, away_team):
    """
    Retrieves past head-to-head results between two teams from the dataset.

    Args:
        df (pd.DataFrame): The game data.
        home_team (str): Name of the home team.
        away_team (str): Name of the away team.

    Returns:
        pd.DataFrame: DataFrame containing past games between the two teams, or empty DataFrame if no matches found.
    """
    past_games = df[((df['Home_Team'] == home_team) & (df['Away_Team'] == away_team)) |
                   ((df['Home_Team'] == away_team) & (df['Away_Team'] == home_team))]
    return past_games

if __name__ == "__main__":
    # Load and prepare the data
    nba_df = load_and_prepare_data()
    if nba_df is None:
        exit()

    # Models to compare
    models = ['random_forest']
    best_model = None
    best_auc_roc = 0

    # Iterate through models
    for model_name in models:
        # Create the model and pipeline
        pipeline, numerical_features = create_model_and_pipeline(nba_df, model=model_name)
        if pipeline is None:
            continue  # Skip to the next model if pipeline creation failed

        # Train and evaluate the model
        trained_model, report, auc_roc, avg_accuracy, y_test, y_pred, losses, X_test = train_and_evaluate_model(pipeline, nba_df, model_name, numerical_features)
        if trained_model is None:
            continue

        # Plot ROC Curve
        plot_roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1], model_name=model_name)
        plot_confusion_matrix(y_test, y_pred, model_name=model_name)
        plot_loss_curve(losses, model_name=model_name) #plot loss curve

        # Determine the best model based on AUC-ROC
        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_model = trained_model
            best_model_name = model_name

    print(f"\nBest Model: {best_model_name} with AUC-ROC: {best_auc_roc:.2f}")

    # Example: Predict winner given home and away teams using the best model
    if best_model:
        home_team = "Houston Rockets"
        away_team = "Golden State Warriors"
        predicted_probability = predict_winner(best_model, nba_df, home_team, away_team, predict_prob='Home')
        if predicted_probability is not None:
            print(f"Predicted probability of {home_team} winning (using {best_model_name}): {predicted_probability:.2f}")

        # Example: Get past head-to-head results
        past_results = get_past_results(nba_df, home_team, away_team)
        print("\nPast Head-to-Head Results:")
        print(past_results[['Home_Team', 'Away_Team', 'Home_Team_Wins']])
    else:
        print("No model was trained successfully.")