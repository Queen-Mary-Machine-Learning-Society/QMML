from utils.monte_carlo_base import GameMonteCarloBase

class NominalMonteCarlo(GameMonteCarloBase):
    def __init__(self, n_simulations, bet, multiplier, odds=0.55,
                 initial_amount=100, n_bets=30, plot=False, log=True,  threshold=0.50):
        super().__init__(n_simulations, bet, multiplier, odds,
                 initial_amount, n_bets, plot, log, threshold)
    
    def game(self, portfolio):
        for _ in range(self.n_bets):
            outcome = self.return_generator(self.bet)
            profit = outcome - self.bet
            portfolio.append(portfolio[-1]+profit)
            if portfolio[-1] < self.bet:
                return portfolio
        return portfolio

class PercentageMonteCarlo(GameMonteCarloBase):
    def __init__(self, n_simulations, bet, multiplier, odds=0.55,
                 initial_amount=100, n_bets=30, plot=False, log=True, threshold=0.50):
        super().__init__(n_simulations, bet, multiplier, odds,
                 initial_amount, n_bets, plot, log, threshold)
        self.is_percentage = True
    
    def game(self, portfolio):
        assert (self.bet >= 0) and (self.bet <= 1)
        for _ in range(self.n_bets):
            
            bet_size = portfolio[-1]*self.bet
            if bet_size <= self.threshold:
                return portfolio
            outcome = self.return_generator(bet_size)
            profit = outcome - bet_size
            portfolio.append(portfolio[-1]+profit)
            if portfolio[-1] < self.bet:
                return portfolio
            if bet_size <= self.threshold:
                return portfolio
        return portfolio