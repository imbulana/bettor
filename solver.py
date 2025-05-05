import argparse
import numpy as np
from scipy.optimize import minimize, Bounds


def optimize(n, B, trade_off, p_i, o_i):
    """
    Optimize betting strategy using the given parameters.

    Args:
        n (int): Number of bets
        B (float): Total budget for betting
        trade_off (float): Risk-return trade-off parameter
        p_i (np.ndarray): Win probabilities for each bet
        o_i (np.ndarray): Payout odds for each bet 

    Returns:
        bets (np.ndarray): Optimal bet amounts for each bet
        profit (float): Expected profit from the betting strategy
    """

    # coeffs
    linear_coeff = p_i * o_i - 1.0
    quadratic_coeff = trade_off * o_i**2 * p_i * (1.0 - p_i)

    # objective function
    def g(x):
        return -np.dot(linear_coeff, x) + np.dot(quadratic_coeff, x**2)

    # objective function gradient
    def grad_g(x):
        return 2 * quadratic_coeff * x - linear_coeff

    # constraints
    bounds = Bounds(0, np.inf)
    cons = {
        'type': 'ineq', 'fun': lambda x: B - np.sum(x), 'jac': lambda x: -np.ones_like(x)
    }

    # initial guess (distribute budget evenly)
    x0 = np.ones(n)*(B/n)

    # solve
    res = minimize(
        fun=g,
        x0=x0,
        jac=grad_g,
        bounds=bounds,
        constraints=cons,
        method='SLSQP',
        options={'ftol':1e-9,} #'disp':True}
    )

    bets = res.x
    profit = np.dot(linear_coeff, bets)

    return bets, profit

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--budget', type=float, help='Total budget for betting'
    )
    parser.add_argument(
        '--trade-off', type=float, default=0.1, help='Risk-return trade-off parameter'
    )
    parser.add_argument(
        '--win-probs', type=float, nargs='+', help='Win probabilities for each bet'
    )
    parser.add_argument(
        '--payout-odds', type=float, nargs='+', help='Payout odds for each bet'
    )

    args = parser.parse_args()
    
    win_probs = np.array(args.win_probs)
    payout_odds = np.array(args.payout_odds)
    
    if len(win_probs) != len(payout_odds):
        raise ValueError("The number of win probabilities must match the number of payout odds")
    
    n_bets = len(win_probs)
    budget = args.budget
    trade_off = args.trade_off
    
    bets, profit = optimize(n_bets, budget, trade_off, win_probs, payout_odds)

    print(f"Budget: {budget}")
    print("\nBet allocation:")
    for i, amount in enumerate(bets):
        print(f"bet {i+1}: {amount:.2f} (Win prob: {win_probs[i]:.2f}, Odds: {payout_odds[i]:.2f})")
    
    print(f"\nTotal allocated: {np.sum(bets):.2f}")
    print(f"Expected profit: {profit:.2f}")
    
    return bets, profit

if __name__ == "__main__":
    main()