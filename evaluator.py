import sys
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

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

def evaluate_betting_strategy(model, model_data, test_data, budget, trade_off=0.5):

    # load model
    with open(model, 'rb') as f:
        rf_model = pkl.load(f)

    # load model data
    nba_df = pd.read_csv(model_data)
    nba_df.dropna(inplace=True)

    # load test data
    playoff_df = pd.read_csv(test_data) 

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
        if win_prob is None:
            continue        

        choose_home_team = win_prob > 0.5
        home_team_wins = score_home > score_away
        
        p_i.append(win_prob if choose_home_team else 1 - win_prob)
        o_i.append(o_h_i if choose_home_team else o_a_i)
        
        winning_team_chosen.append(choose_home_team == home_team_wins)

    n = len(p_i)
    p_i, o_i = np.array(p_i), np.array(o_i)
    bets, expected_profit = optimize(n, budget, p_i, o_i, trade_off)

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

if __name__ == '__main__':
    model = sys.argv[1] 
    model_data = sys.argv[2]
    test_data = sys.argv[3]
    budget = float(sys.argv[4])

    print(budget)

    trade_offs = [0.25 * i for i in range(1, 10)]
    actual_profits = []
    expected_profits = []
    actual_profits_percentages = []
    bet_amounts = []
    total_bet_amounts = []
    for trade_off in trade_offs:
        result = evaluate_betting_strategy(model, model_data, test_data, budget, trade_off)
        actual_profits.append(result['actual_profit'])
        expected_profits.append(result['expected_profit'])

        total_bet_amount = sum(result['bets'])
        total_bet_amounts.append(total_bet_amount)
        bet_amounts.append(result['bets'])

        # actual_profit_percent = result['actual_profit'] / total_bet_amount * 100
        actual_profit_percent = result['actual_profit'] / budget * 100
        actual_profits_percentages.append(actual_profit_percent)

        n_games = len(result['winning_team_chosen'])
        print(f"\nTrade-off: {trade_off}")
        print(f"number of games: {n_games}")
        print(f"Total bets: {total_bet_amount}")
        print(f"Expected profit: {result['expected_profit']}")
        print(f"Actual profit: {result['actual_profit']}")
        print(f"Actual profit percentage: {actual_profit_percent:.2f}%")
        print(f"winning bet percentage: {sum(result['winning_team_chosen'])/n_games}")

    winning_percentage = sum(result['winning_team_chosen']) / len(result['winning_team_chosen']) * 100

    # plotting actual profit percentage vs trade-off parameter
    plt.figure(figsize=(10, 6))
    plt.plot(trade_offs, actual_profits_percentages, label='Actual Profit Gain', marker='o')

    for i, txt in enumerate(total_bet_amounts):
        plt.annotate(
            f'${txt:.0f}', 
            (trade_offs[i], plt.ylim()[0]),
            xytext=(0, -18),
            textcoords='offset points',
            ha='center',
            va='top'
        )

    plt.xlabel('Trade-off Parameter and Total Bet Amount', labelpad=20)
    plt.ylabel('Profit Percentage')
    plt.title(
        f'Profit Gain vs. Trade off parameter (Budget: ${budget}, winning bet percentage: {winning_percentage:.2f}%)'
    )
    plt.legend()
    plt.grid(True)
    plt.show()

    # plotting actual vs expected profit
    plt.figure(figsize=(10, 6))
    plt.plot(trade_offs, actual_profits, label='Actual Profit', marker='o')
    plt.plot(trade_offs, expected_profits, label='Expected Profit', marker='x')

    for i, txt in enumerate(total_bet_amounts):
        plt.annotate(
            f'${txt:.0f}', 
            (trade_offs[i], plt.ylim()[0]),
            xytext=(0, -18),
            textcoords='offset points',
            ha='center',
            va='top'
        )

    plt.xlabel('Trade-off Parameter and Total Bet Amount', labelpad=20)
    plt.ylabel('Profit ($)')
    plt.title(f'Actual vs Expected Profit for Different Trade-off Parameters (Budget: $1000, winning bet percentage: {winning_percentage:.2f}%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plotting bet amount distribution
    plt.figure(figsize=(10, 6))
    plt.violinplot(bet_amounts)
    plt.boxplot(
        bet_amounts, 
        labels=[f'{t:.1f}' for t in trade_offs], 
        widths=0.3,
        patch_artist=True, 
        boxprops=dict(facecolor='white', alpha=0.7),
        medianprops=dict(color='red')
    )
    
    plt.xlabel('Trade-off Parameter')
    plt.ylabel('Bet Amount ($)')
    plt.title('Distribution of Bet Amounts')
    plt.grid(True)
    plt.show()