import pandas as pd
import pickle as pkl

from model import predict_winner
from solver import optimize

team_name = {
    'atl': 'Atlanta Hawks',
    'bos': 'Boston Celtics',
    'bkn': 'Brooklyn Nets',
    'cha': 'Charlotte Hornets',
    'chi': 'Chicago Bulls',
    'cle': 'Cleveland Cavaliers',
    'dal': 'Dallas Mavericks',
    'den': 'Denver Nuggets',
    'det': 'Detroit Pistons',
    'gs': 'Golden State Warriors',
    'hou': 'Houston Rockets',
    'ind': 'Indiana Pacers',
    'lac': 'Los Angeles Clippers',
    'lal': 'Los Angeles Lakers',
    'mem': 'Memphis Grizzlies',
    'mia': 'Miami Heat',
    'mil': 'Milwaukee Bucks',
    'min': 'Minnesota Timberwolves',
    'no': 'New Orleans Pelicans',
    'ny': 'New York Knicks',
    'okc': 'Oklahoma City Thunder',
    'orl': 'Orlando Magic',
    'phi': 'Philadelphia 76ers',
    'phx': 'Phoenix Suns',
    'por': 'Portland Trail Blazers',
    'sac': 'Sacramento Kings',
    'sa': 'San Antonio Spurs',
    'tor': 'Toronto Raptors',
    'utah': 'Utah Jazz',
    'wsh': 'Washington Wizards'
}

def evaluate_betting_strategy(budget):

    # load model
    with open('random_forest_model.pkl', 'rb') as f:
        rf_model = pkl.load(f)

    # load regular season data
    nba_df = pd.read_csv('data/total_data.csv')
    nba_df.dropna(inplace=True)

    # load playoff data
    playoff_df = pd.read_csv('data/nba_model_inputs.csv') 
    playoff_df = playoff_df[playoff_df['playoff'] == True]

    p_i = [] # win probabilities
    o_i = [] # payout odds

    winning_team_chosen = [] # does actual winner match with the betting strategy

    for home, away, o_h_i, o_a_i, score_home, score_away in zip(
        playoff_df['home'], playoff_df['away'], 
        playoff_df['oi_home'], playoff_df['oi_away'],
        playoff_df['score_home'], playoff_df['score_away']
    ):
        win_prob = predict_winner(
            rf_model, nba_df, team_name[home], team_name[away], predict_prob='Home'
        )
        
        choose_home_team = win_prob > 0.5
        home_team_wins = score_home > score_away
        
        p_i.append(win_prob if choose_home_team else 1 - win_prob)
        o_i.append(o_h_i if choose_home_team else o_a_i)
        
        winning_team_chosen.append(choose_home_team == home_team_wins)

    n = len(p_i)
    # TODO: add trade_off factor
    bets, expected_profit = optimize(n, budget, p_i, o_i)

    # calculate actual profit from betting strategy
    actual_profit = 0
    for i in range(n):
        if winning_team_chosen[i]:
            actual_profit += bets[i] * (o_i[i] - 1)
        else:
            actual_profit -= bets[i]

    return {
        'bets': bets,
        'actual_profit': actual_profit,
        'expected_profit': expected_profit,
        'winning_team_chosen': winning_team_chosen,
    }