import pandas as pd

# Load data
games = pd.read_csv('data/games.csv')
teams = pd.read_csv('data/teams.csv')

# Keep only team_id and abbreviation
teams = teams[['TEAM_ID', 'ABBREVIATION']]

# Merge home team abbreviation
games = games.merge(teams, how='left', left_on='HOME_TEAM_ID', right_on='TEAM_ID')
games.rename(columns={'ABBREVIATION': 'HOME_TEAM'}, inplace=True)

# Merge away team abbreviation
games = games.merge(teams, how='left', left_on='VISITOR_TEAM_ID', right_on='TEAM_ID')
games.rename(columns={'ABBREVIATION': 'AWAY_TEAM'}, inplace=True)

# Desired order: abbreviations first, then the rest
games = games[['HOME_TEAM', 'AWAY_TEAM'] + [col for col in games.columns if col not in ['HOME_TEAM_ABBR', 'AWAY_TEAM_ABBR']]]
games.drop(columns=['GAME_STATUS_TEXT'], inplace=True)

games.to_csv('data/games_cleaned.csv', index=False)