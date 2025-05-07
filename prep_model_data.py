import pandas as pd

# Load filtered dataset
df = pd.read_csv('data/nba_odds_2022_2023_filtered.csv')

# Define moneyline to decimal conversion
def moneyline_to_decimal(ml):
    return (ml / 100 + 1) if ml > 0 else (100 / abs(ml)) + 1

# Convert to decimal payout odds
df['oi_home'] = df['moneyline_home'].apply(moneyline_to_decimal)
df['oi_away'] = df['moneyline_away'].apply(moneyline_to_decimal)

# Estimate implied win probabilities
df['pi_home'] = 1 / df['oi_home']
df['pi_away'] = 1 / df['oi_away']

# Normalize for bookmaker margin
df['pi_home'] /= (df['pi_home'] + df['pi_away'])
df['pi_away'] = 1 - df['pi_home']

# Save final model-ready format
df[['date', 'playoffs', 'home', 'away', 'score_home', 'score_away', 'pi_home', 'oi_home', 'pi_away', 'oi_away']].to_csv(
    'data/nba_model_inputs.csv', index=False)

print("Model-ready file saved as nba_model_inputs.csv")