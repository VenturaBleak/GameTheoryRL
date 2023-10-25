import torch
from game_env import GameEnv
from model import DQN
from tqdm import tqdm

from strategies import StrategyFactory
from hyperparameters import Hyperparameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyperparameter = Hyperparameter()
NUM_EPISODES = hyperparameter.NUM_EVAL_EPISODES
MAX_ROUNDS_PER_EPISODE = hyperparameter.MAX_ROUNDS_PER_EPISODE

env = GameEnv("prisoners_dilemma", render_mode=None, max_rounds=MAX_ROUNDS_PER_EPISODE)
input_dim = len(env.reset()[0][0])
output_dim = env.action_space.n
hidden_sizes = hyperparameter.HIDDEN_SIZES

model_agent1 = DQN(input_dim, hidden_sizes, output_dim).to(device)
model_agent1.load_state_dict(torch.load(f"{env.game_modes.game_type}_agent1.pth"))
model_agent1.eval()

# List of strategies
strategies_list = list(StrategyFactory.strategy_probabilities.keys())
results = {}
strategy_scores = {strategy: 0 for strategy in strategies_list}  # For storing total scores of each strategy

# Tournament loop
for strat1 in tqdm(strategies_list):
    for strat2 in strategies_list:
        env = GameEnv("prisoners_dilemma", render_mode=None, max_rounds=MAX_ROUNDS_PER_EPISODE)
        # Initialize results storage
        results[(strat1, strat2)] = {"agent1_reward": 0, "agent2_reward": 0}

        strategy_agent1 = StrategyFactory.create_strategy(env, strategy_type=strat1)
        strategy_agent2 = StrategyFactory.create_strategy(env, strategy_type=strat2)

        for _ in range(NUM_EPISODES):
            observations, _ = env.reset()
            observation_agent1, observation_agent2 = observations
            for _ in range(MAX_ROUNDS_PER_EPISODE+1):
                action_agent1 = strategy_agent1.select_action(observation_agent1, model_agent1)
                action_agent2 = strategy_agent2.select_action(observation_agent2, model_agent1)

                observations, _, terminated, _, _ = env.step(action_agent1, action_agent2)
                observation_agent1, observation_agent2 = observations

                if terminated:
                    env.episode_counter += 1
                    break

            # Accumulate rewards
            results[(strat1, strat2)]["agent1_reward"] += env.agent1_total_cumulative_reward
            results[(strat1, strat2)]["agent2_reward"] += env.agent2_total_cumulative_reward
            # Add rewards to strategy_scores for later calculation of averages
            strategy_scores[strat1] += env.agent1_total_cumulative_reward
            strategy_scores[strat2] += env.agent2_total_cumulative_reward

# Calculate average scores for each strategy
num_strategies = len(strategies_list)
for strategy in strategy_scores:
    strategy_scores[strategy] /= (num_strategies * NUM_EPISODES)

# Sort strategies by average scores
sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)

# Print results
for key, value in results.items():
    print(f"Strategy Pair: {key[0]} vs {key[1]}")
    print(f"Agent 1 (Strategy: {key[0]}) Average Cumulative Reward: {value['agent1_reward'] / NUM_EPISODES}")
    print(f"Agent 2 (Strategy: {key[1]}) Average Cumulative Reward: {value['agent2_reward'] / NUM_EPISODES}")
    print("---------------------------------------------------")

print("\nOverall Strategy Rankings:")
for i, (strategy, score) in enumerate(sorted_strategies, 1):
    print(f"{i}. {strategy}: {score}")