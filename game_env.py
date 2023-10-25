import gymnasium as gym
from gymnasium import spaces
import numpy as np

from game_modes import GameModes
from game_renderer import GameRenderer

class GameEnv(gym.Env):
    def __init__(self, game_type, render_mode, max_rounds=10):
        super(GameEnv, self).__init__()

        self.game_modes = GameModes(game_type)
        num_actions = len(self.game_modes.get_action_names())
        min_payoff, max_payoff = self.game_modes.get_payoff_bounds()

        self.terminated = False
        self.truncated = False

        self.action_space = spaces.Discrete(num_actions)
        self.render_mode = render_mode
        self.max_rounds = max_rounds
        self.current_round = 0
        self.total_rounds = 0
        self.episode_counter = 1  # Initialize episode counter

        # Initialize static structures with placeholders
        self.actions_history = np.zeros((self.max_rounds, 2), dtype=int)
        self.rewards_history = np.full((self.max_rounds, 2), 0)

        # Masks initialized with zeros
        self.actions_mask = np.zeros(self.max_rounds, dtype=int)
        self.payoffs_mask = np.zeros((self.max_rounds, 2), dtype=int)

        # observation space
        self.observation_space = spaces.Tuple([
            spaces.MultiBinary([num_actions, num_actions] * max_rounds),  # Actions history
            spaces.MultiBinary(max_rounds),  # Actions mask
            # spaces.Discrete(max_rounds),  # Current round
            # spaces.Discrete(max_rounds),  # Total rounds
            # spaces.Box(low=min_payoff, high=max_payoff, shape=(num_actions ** 2,), dtype=np.float32)  # Game params
        ])

        self.renderer = None
        if render_mode == 'human':
            self.renderer = GameRenderer(self.game_modes)  # Pass instantiated GameModes object to GameRenderer

        # reset after each step
        self.agent1_action = None
        self.agent2_action = None
        self.agent1_reward = 0
        self.agent2_reward = 0

        # reset after each episode
        self.agent1_cumulative_reward = 0
        self.agent2_cumulative_reward = 0
        self.agent1_avg_reward = 0
        self.agent2_avg_reward = 0

        # never reset
        self.agent1_total_cumulative_reward = 0
        self.agent2_total_cumulative_reward = 0
        self.agent1_total_avg_reward = 0
        self.agent2_total_avg_reward = 0

    def reset(self):
        self.current_round = 0

        # Fill history with placeholders
        self.actions_history = np.zeros((self.max_rounds, 2), dtype=int)
        self.actions_mask = np.zeros(self.max_rounds, dtype=int)
        self.rewards_history.fill(0)

        self.agent1_avg_reward = 0
        self.agent2_avg_reward = 0

        self.agent1_cumulative_reward = 0
        self.agent2_cumulative_reward = 0

        return (self._get_observation(1), self._get_observation(2)), {}

    def step(self, action_agent1, action_agent2):
        self.current_round += 1
        self.total_rounds += 1

        # get actions from both agents
        self.agent1_action = action_agent1
        self.agent2_action = action_agent2  # This is our opponent for agent1

        # get payoffs from game
        payoff = self.game_modes.get_payoff(self.agent1_action, self.agent2_action)
        self.agent1_reward = payoff[0]
        self.agent2_reward = payoff[1]

        # Store actions and payoffs in history
        self.actions_history[self.current_round - 1] = [action_agent1, action_agent2]
        self.actions_mask[self.current_round - 1] = 1
        self.rewards_history[self.current_round - 1] = [self.agent1_reward, self.agent2_reward]

        self.terminated = self.current_round >= self.max_rounds

        obs_agent1 = self._get_observation(1)
        obs_agent2 = self._get_observation(2)

        # episode reward
        self.agent1_cumulative_reward += self.agent1_reward
        self.agent2_cumulative_reward += self.agent2_reward
        self.agent1_avg_reward = self.agent1_cumulative_reward / self.current_round
        self.agent2_avg_reward = self.agent2_cumulative_reward / self.current_round

        # total reward
        self.agent1_total_cumulative_reward += self.agent1_reward
        self.agent2_total_cumulative_reward += self.agent2_reward
        self.agent1_total_avg_reward = self.agent1_total_cumulative_reward / self.total_rounds
        self.agent2_total_avg_reward = self.agent2_total_cumulative_reward / self.total_rounds

        # print for a sanity check
        # print(f"mask: {self.actions_mask}")
        # print(f"history: {self.actions_history}")

        return (obs_agent1, obs_agent2), (payoff[0], payoff[1]), self.terminated, self.truncated, {}

    def _get_observation(self, agent_number):
        if agent_number == 1:
            actions_hist = self.actions_history
        else:
            actions_hist = np.column_stack((self.actions_history[:, 1], self.actions_history[:, 0]))

        observation = self.flatten_observation(actions_hist, self.actions_mask)

        return observation

    def flatten_observation(self, actions_hist, actions_mask):
        flat_list = list(actions_hist.flatten()) + list(actions_mask)
        return flat_list

    def render(self):
        if self.render_mode == 'human':
            self.renderer.render(self.agent1_action, self.agent2_action, self.agent1_reward, self.agent2_reward,
                                 self.agent1_cumulative_reward, self.agent2_cumulative_reward,
                                 self.agent1_avg_reward, self.agent2_avg_reward,
                                 self.agent1_total_cumulative_reward, self.agent2_total_cumulative_reward,
                                 self.agent1_total_avg_reward, self.agent2_total_avg_reward,
                                 self.episode_counter,
                                 self.current_round)

    def close(self):
        self.renderer.close()