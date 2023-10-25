"""
Hyperparameters for the game theory model.
"""

# class
class Hyperparameter:
    def __init__(self):

        # Game parameters
        self.NUM_EPISODES = 10000
        self.MAX_ROUNDS_PER_EPISODE = 10
        self.NUM_EVAL_EPISODES = 1000

        # Model parameters
        self.REPLAY_BUFFER_SIZE = int(self.NUM_EPISODES * self.MAX_ROUNDS_PER_EPISODE * 0.05)
        self.HIDDEN_SIZES = [128, 128, 128, 128, 128]
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.001
        self.GAMMA = 0.995  # Discount factor for future rewards

        # Epsilon-greedy action selection
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = self.NUM_EPISODES * 0.8  # Decaying length for epsilon-greedy action selection

        # Self play parameters
        self.TARGET_UPDATE = 10
        self.SELF_PLAY_UPDATE = 50
