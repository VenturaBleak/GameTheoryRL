import numpy as np

class GameModes:
    def __init__(self, game_type):
        self.game_type = game_type  # Store game_type as an attribute
        self.games = {
            "rock_paper_scissors": {
                "payoffs": np.array([[(0, 0), (-1, 1), (1, -1)],
                                     [(1, -1), (0, 0), (-1, 1)],
                                     [(-1, 1), (1, -1), (0, 0)]]),
                "action_names": ["rock", "paper", "scissors"]
            },
            "chicken": {
                "payoffs": np.array([[(0, 0), (-1, 1)],
                                     [(1, -1), (-10, -10)]]),
                "action_names": ["go_straight", "swerve"]
            },
            "battle_of_sexes": {
                "payoffs": np.array([(3, 2), (1, 1), (0, 0), (2, 3)]),
                "action_names": ["opera", "football"]
            },
            "matching_pennies": {
                "payoffs": np.array([(1, -1), (-1, 1), (-1, 1), (1, -1)]),
                "action_names": ["heads", "tails"]
            },
            "prisoners_dilemma": {
                "payoffs": np.array([[(3, 3), (0, 5)], [(5, 0), (1, 1)]]),
                "action_names": ["cooperate", "defect"]
            }
        }

    def get_payoff(self, action1, action2):
        return self.games[self.game_type]["payoffs"][action1][action2]

    def get_payoff_matrix(self):
        return self.games[self.game_type]["payoffs"]

    def get_payoff_bounds(self):
        # Fetching the payoffs matrix for the current game type
        payoff_matrix = self.games[self.game_type]["payoffs"]
        min_payoff = np.min(payoff_matrix)
        max_payoff = np.max(payoff_matrix)
        return min_payoff, max_payoff

    def get_action_names(self):
        return self.games[self.game_type]["action_names"]

    def get_game(self):
        return self.games.get(self.game_type, None)