import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import sys

from solver import optimize

team_name = {
    'atl': 'ATL',
    'bos': 'BOS',
    'bkn': 'BKN',
    'cha': 'CHA',
    'chi': 'CHI',
    'cle': 'CLE',
    'dal': 'DAL',
    'den': 'DEN',
    'det': 'DET',
    'gs': 'GSW',
    'hou': 'HOU',
    'ind': 'IND',
    'lac': 'LAC',
    'lal': 'LAL',
    'mem': 'MEM',
    'mia': 'MIA',
    'mil': 'MIL',
    'min': 'MIN',
    'no': 'NOP',
    'ny': 'NYK',
    'okc': 'OKC',
    'orl': 'ORL',
    'phi': 'PHI',
    'phx': 'PHX',
    'por': 'POR',
    'sac': 'SAC',
    'sa': 'SAS',
    'tor': 'TOR',
    'utah': 'UTA',
    'wsh': 'WAS'
}

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pkl.load(f)
    return model

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def evaluate_betting_strategy(win_probs, test_data, budget, trade_off=0.5):

    def get_win_prob(home_team, away_team):
        return win_probs[
            (win_probs['home_team'] == home_team) & (win_probs['away_team'] == away_team)
        ]['home_win_prob'].iloc[0]

    winning_team_chosen = []
    p_i = []
    o_i = []
    for i, row in test_data.iterrows():
        home_team = team_name[row['home']]
        away_team = team_name[row['away']]
        score_home = row['score_home']
        score_away = row['score_away']
        oi_home = row['oi_home']
        oi_away = row['oi_away']

        win_prob = get_win_prob(home_team, away_team)
        if win_prob is None:
            print(f"Win probability not found for {home_team} vs {away_team}")
            continue
        choose_home_team = win_prob > 0.5
        home_team_wins = score_home > score_away
        
        p_i.append(win_prob if choose_home_team else 1 - win_prob)
        o_i.append(oi_home if choose_home_team else oi_away)
        
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

if __name__ == "__main__":
    win_probs_path = sys.argv[2] if len(sys.argv) > 2 else 'win_prob.csv'
    test_data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/nba_22_23_jan.csv'

    win_probs = load_data(win_probs_path)
    test_data = load_data(test_data_path)

    budget = 10_000
    print(f'budget: {budget}')

    trade_offs = [i/100 for i in range(1, 11)]
    # trade_offs = [300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
    # trade_offs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # trade_offs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    actual_profits = []
    actual_profits_percentages = []
    expected_profits = []
    total_bet_amounts = []
    bet_amounts = []

    for trade_off in trade_offs:
        result = evaluate_betting_strategy(win_probs, test_data, budget, trade_off)
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