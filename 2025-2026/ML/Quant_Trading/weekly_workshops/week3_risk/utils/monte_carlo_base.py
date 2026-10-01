"""
Hidden challenge:
The following implementation can be optimized by using vectorization with numpy.
For an example, refer to tutoirals/algorithmic_trading
"""
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod


class GameMonteCarloBase(ABC):
    def __init__(self, n_simulations, bet, multiplier, odds=0.55,
                 initial_amount=100, n_bets=30, plot=False, log=True,
                 threshold = 0.5):
        self.n_simulations = n_simulations
        self.bet = bet
        self.multiplier = multiplier
        self.odds = odds
        self.initial_amount = initial_amount
        self.n_bets = n_bets
        self.plot = plot
        self.log = log
        self.threshold = threshold
        self.is_percentage = False
    
    def return_generator(self, bet_size):
        rng = np.random.default_rng()
        outcome = rng.uniform(0,1)
        if self.odds > outcome:
            return self.multiplier * bet_size + bet_size
        else:
            return 0
    
    @abstractmethod
    def game(self):
        pass
    
    def _is_broke(self, portfolio):
        if self.is_percentage:
            return portfolio[-1] < self.initial_amount * self.threshold
        else:
            return portfolio[-1] < self.bet
        
    def monte_carlo_simulation(self):
        final_array = []
        profit = []
        fails = 0 # Count how many times we run out of money
        for _ in range(self.n_simulations):
            portfolio = [self.initial_amount]
            portfolio = self.game(portfolio=portfolio)
            if self._is_broke(portfolio):
                fails += 1
            strategy_return = portfolio[-1]/self.initial_amount - 1
            profit.append(strategy_return)
            final_array.append(portfolio)
        
        if self.log:
            print(f"\nMean return of the strategy: {np.mean(profit)*100}%")
            print(f"\nHow often we run out of money: {fails/self.n_simulations*100}%")

        if self.plot:

            plt.figure(figsize=(10,6))
            for array in final_array:
                plt.plot(array)
            plt.grid(True)
            plt.title("Monte Carlo Simulation of the game")
            plt.ylabel("Profit Paths")
            plt.xlabel("Number of bets")
            plt.show()

        return np.mean([p[-1] for p in final_array]), fails/self.n_simulations
    