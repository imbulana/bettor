import pandas as pd

<<<<<<< HEAD
# Load filtered dataset for 2022–23
df = pd.read_csv('nba_odds_2022_2023_filtered.csv')
=======
# Load filtered dataset
df = pd.read_csv('data/nba_odds_2022_2023_filtered.csv')
>>>>>>> 2d298e7e08d3f6367185d1bef404f664153fd62b

# Drop rows where odds are still missing (just in case)
df = df[df['moneyline_home'].notnull() & df['moneyline_away'].notnull()]

# Convert moneyline to decimal payout odds
def moneyline_to_decimal(ml):
    return (ml / 100 + 1) if ml > 0 else (100 / abs(ml)) + 1

df['oi_home'] = df['moneyline_home'].apply(moneyline_to_decimal)
df['oi_away'] = df['moneyline_away'].apply(moneyline_to_decimal)

# Estimate implied win probabilities
df['pi_home'] = 1 / df['oi_home']
df['pi_away'] = 1 / df['oi_away']

# Normalize for bookmaker margin
df['pi_home'] /= (df['pi_home'] + df['pi_away'])
df['pi_away'] = 1 - df['pi_home']

# Save the model-ready dataset
df[['date', 'home', 'away', 'pi_home', 'oi_home', 'pi_away', 'oi_away']].to_csv(
    'nba_betting_odds_2021_2022.csv', index=False)

print(f"Model-ready file saved with {len(df)} games (2021–22 season)")