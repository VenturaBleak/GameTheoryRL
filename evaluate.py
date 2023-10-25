import torch
from tqdm import trange

from model import DQN
from game_env import GameEnv
from strategies import StrategyFactory
from hyperparameters import Hyperparameter

##############################################################################################################
# Evaluate the model
##############################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyperparameter = Hyperparameter()
NUM_EPISODES = hyperparameter.NUM_EVAL_EPISODES
MAX_ROUNDS_PER_EPISODE = hyperparameter.MAX_ROUNDS_PER_EPISODE

env = GameEnv("prisoners_dilemma", render_mode=None, max_rounds=MAX_ROUNDS_PER_EPISODE)
input_dim = len(env.reset()[0][0])
output_dim = env.action_space.n

# Use the same hidden layer structure as in train.py
hidden_sizes = hyperparameter.HIDDEN_SIZES

# Adjust the model instantiation
model_agent1 = DQN(input_dim, hidden_sizes, output_dim).to(device)
model_agent1.load_state_dict(torch.load(f"{env.game_modes.game_type}_agent1.pth"))
model_agent1.eval()  # Set to evaluation mode

# select opponent strategy
strategy_agent1 = StrategyFactory.create_strategy(env, strategy_type="model")
strategy_agent2 = StrategyFactory.create_strategy(env, strategy_type="sample_from_dict")

##############################################################################################################
# Evaluate the model
##############################################################################################################
for _ in trange(NUM_EPISODES):
    observations, _ = env.reset()
    observation_agent1, observation_agent2 = observations
    for _ in range(MAX_ROUNDS_PER_EPISODE + 1):
        action_agent1 = strategy_agent1.select_action(observation_agent1, model_agent1)
        action_agent2 = strategy_agent2.select_action(observation_agent2, model_agent1)

        observations, _, terminated, _, _ = env.step(action_agent1, action_agent2)

        observation_agent1, observation_agent2 = observations

        # # # print for a sanity check
        # print(f"-------------------")
        # print(f"current_round: {env.current_round}")
        # print(f"sampled strategy: {strategy_agent2.__class__.__name__}")
        # print(f"observation_agent2: {observation_agent2}")
        # print(f"action_agent1 previous round (opponent): {env.actions_history[env.current_round - 1][1]}")
        # print(f"action_agent2: {action_agent2}")

        if terminated:
            env.episode_counter += 1  # Increment episode counter after termination
            break

print(f"Agent 1 Total Cumulative Reward: {env.agent1_total_cumulative_reward}")
print(f"Agent 2 Total Cumulative Reward: {env.agent2_total_cumulative_reward}")

##############################################################################################################
# Visualize the game
##############################################################################################################

env = GameEnv("prisoners_dilemma", render_mode="human", max_rounds=MAX_ROUNDS_PER_EPISODE)
strategy_agent2 = StrategyFactory.create_strategy(env, strategy_type="sample_from_dict")

NUM_EPISODES = 100

for _ in range(NUM_EPISODES):
    observations, _ = env.reset()
    observation_agent1, observation_agent2 = observations
    for _ in range(MAX_ROUNDS_PER_EPISODE):
        action_agent1 = strategy_agent1.select_action(observation_agent1, model_agent1)
        action_agent2 = strategy_agent2.select_action(observation_agent2, model_agent1)
        observations, _, terminated, _, _ = env.step(action_agent1, action_agent2)
        observation_agent1, observation_agent2 = observations

        env.render()
        if terminated:
            env.episode_counter += 1  # Increment episode counter after termination
            break