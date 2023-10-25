"""
Multi-Agent Deep Q-learning for Game Theory like games.
    1. The game progresses in episodes, with each episode consisting of several rounds.
    2. In each round, agents make decisions based on their current understanding of the game, and the results of those decisions are stored in the replay memory.
    3. The agents then learn from these stored experiences.
    4. Every so often (as defined by TARGET_UPDATE), the DQNs' target networks are updated to ensure stable learning.
    And periodically (as defined by SELF_PLAY_UPDATE), agent 2's strategy is updated to match agent 1's, ensuring that both agents are learning from and adapting to each other in a self-play setup.

TARGET_UPDATE -> stabilizes learning through keeping step q-values and target q-values separate
1. Policy Network:
    Role:
    This network is used to decide the action to take at every step, i.e., it's the one that's actively being trained to
    improve its strategy over time.

    Updates:
    It gets updated frequently (every time we decide to perform backpropagation based on the batch sampled from the
     replay memory).


2. Target Network:
    Role:
    This network is used to compute the target Q-values for the Q-learning update. Its purpose is to provide
    stability during training. If we were to use just one network, the Q-learning update would be using current
    estimates to update current estimates. This can lead to harmful feedback loops and unstable training.

    Updates:
    It gets updated less frequently. It's periodically updated to have the same weights as the policy network,
    but between these updates, its weights remain frozen. This periodicity is determined by the TARGET_UPDATE
    parameter. So every TARGET_UPDATE episodes, we copy the weights from the policy network to the target network.

SELF_PLAY_UPDATE: -> keeps the two agents in sync
    Self-play is a methodology where an agent learns by playing against itself. This can lead to
    more robust strategies since the agent is constantly trying to adapt to its own evolving strategy. In the given
    context, after every SELF_PLAY_UPDATE episodes, the strategy (weights of the DQN) of agent 1 is copied over to
    agent 2. This ensures that agent 2 is always trying to adapt to the latest strategy of agent 1. As a result,
    both agents iteratively refine their policies in light of the other's behavior.
"""

# Necessary libraries
from game_env import GameEnv
import torch
import torch.nn.functional as F
from torch import optim
import random
import math
from tqdm import trange

from model import DQN
from buffer import ReplayMemory
from hyperparameters import Hyperparameter

# Training parameters
hyperparameter = Hyperparameter()
NUM_EPISODES = hyperparameter.NUM_EPISODES
MAX_ROUNDS_PER_EPISODE = hyperparameter.MAX_ROUNDS_PER_EPISODE
REPLAY_MEMORY_SIZE = int(NUM_EPISODES * MAX_ROUNDS_PER_EPISODE * 0.05)

# Hyperparameters
BATCH_SIZE = hyperparameter.BATCH_SIZE
GAMMA = hyperparameter.GAMMA  # Discount factor for future rewards
LEARNING_RATE = hyperparameter.LEARNING_RATE

# Target network is a mechanism in DQN to stabilize learning.
# The idea is to keep a separate network (with same architecture as the primary DQN) which is not updated as frequently as the primary DQN.
TARGET_UPDATE = hyperparameter.TARGET_UPDATE

# Every SELF_PLAY_UPDATE episodes, agent 2's policy is updated to that of agent 1.
# This is the essence of self-play where one agent constantly adapts to the policy of the other.
SELF_PLAY_UPDATE = hyperparameter.SELF_PLAY_UPDATE

# Epsilon-greedy strategy parameters
EPS_START = hyperparameter.EPS_START  # Starting value of epsilon
EPS_END = hyperparameter.EPS_END  # Ending value of epsilon
EPS_DECAY = hyperparameter.EPS_DECAY  # Rate at which epsilon decays
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment setup
env = GameEnv("prisoners_dilemma", render_mode=None, max_rounds=MAX_ROUNDS_PER_EPISODE)

# Observations and action spaces
input_dim = len(env.reset()[0][0])
output_dim = env.action_space.n

# Neural network configurations
hidden_sizes = hyperparameter.HIDDEN_SIZES
policy_net1 = DQN(input_dim, hidden_sizes, output_dim).to(device)  # Primary DQN for agent 1
target_net1 = DQN(input_dim, hidden_sizes, output_dim).to(device)  # Target DQN for agent 1
policy_net2 = DQN(input_dim, hidden_sizes, output_dim).to(device)  # Primary DQN for agent 2
target_net2 = DQN(input_dim, hidden_sizes, output_dim).to(device)  # Target DQN for agent 2

# Initialize target network weights with policy network's weights
target_net1.load_state_dict(policy_net1.state_dict())
target_net2.load_state_dict(policy_net2.state_dict())

# Optimizers to adjust neural network weights
optimizer1 = optim.Adam(policy_net1.parameters(), lr=LEARNING_RATE)
optimizer2 = optim.Adam(policy_net2.parameters(), lr=LEARNING_RATE)

# Replay memories to store experiences
memory1 = ReplayMemory(REPLAY_MEMORY_SIZE)
memory2 = ReplayMemory(REPLAY_MEMORY_SIZE)


# Action selection function based on epsilon-greedy strategy
def select_action(state, policy_net, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(output_dim)]], device=device, dtype=torch.long)

# Main training loop
steps_done = 0
for episode in trange(NUM_EPISODES):
    observations, _ = env.reset()  # Reset environment at the start of an episode
    observation_agent1, observation_agent2 = observations

    # Episode loop
    for _ in range(MAX_ROUNDS_PER_EPISODE):
        # Convert observations to tensors
        state_agent1 = torch.tensor(observation_agent1, device=device, dtype=torch.float32).unsqueeze(0)
        state_agent2 = torch.tensor(observation_agent2, device=device, dtype=torch.float32).unsqueeze(0)

        # Select actions for both agents
        action_agent1 = select_action(state_agent1, policy_net1, steps_done)
        action_agent2 = select_action(state_agent2, policy_net2, steps_done)
        steps_done += 1

        # Take a step in the environment using selected actions
        observations, (reward_agent1, reward_agent2), terminated, _, _ = env.step(action_agent1.item(),
                                                                                  action_agent2.item())
        next_observation_agent1, next_observation_agent2 = observations

        # Store transitions into the replay memories
        next_state_agent1 = torch.tensor(next_observation_agent1, device=device, dtype=torch.float32).unsqueeze(0)
        next_state_agent2 = torch.tensor(next_observation_agent2, device=device, dtype=torch.float32).unsqueeze(0)

        # # print, for a sanity check
        # print(f"round: {env.current_round}")
        # print(f"observation_agent1: {observation_agent1}")
        # print(f"observation_agent2: {observation_agent2}")

        memory1.push(state_agent1, action_agent1, next_state_agent1, torch.tensor([reward_agent1], device=device))
        memory2.push(state_agent2, action_agent2, next_state_agent2, torch.tensor([reward_agent2], device=device))

        # Optimization function for the Deep Q-learning algorithm
        def optimize_model(policy_net, target_net, memory, optimizer):
            # Make sure enough experiences have been stored before learning begins
            if len(memory) < BATCH_SIZE:
                return

            # Sample experiences from memory
            transitions = memory.sample(BATCH_SIZE)
            batch = memory1.Transition(*zip(*transitions))

            # Identify states that are not the final state of an episode
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                          dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute the current Q-values
            state_action_values = policy_net(state_batch).gather(1, action_batch)

            # Compute the expected Q-values
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # Compute the Huber Loss between current and expected Q-values
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the policy network
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

        # Perform optimization steps for both agents
        optimize_model(policy_net1, target_net1, memory1, optimizer1)
        optimize_model(policy_net2, target_net2, memory2, optimizer2)

        # Update current observation for the next loop iteration
        observation_agent1, observation_agent2 = next_observation_agent1, next_observation_agent2

        # Every TARGET_UPDATE episodes, update the weights of the target networks
        # This is done to stabilize the learning process. The target network provides a stable target for the primary network to learn.
        if episode % TARGET_UPDATE == 0:
            target_net1.load_state_dict(policy_net1.state_dict())
            target_net2.load_state_dict(policy_net2.state_dict())

        # Every SELF_PLAY_UPDATE episodes, synchronize the policy of agent 2 with that of agent 1.
        # This allows the two agents to constantly adapt to each other's strategies in a self-play setting.
        if episode % SELF_PLAY_UPDATE == 0:
            policy_net2.load_state_dict(policy_net1.state_dict())

        # Check if the episode has terminated
        if terminated:
            env.episode_counter += 1  # Increment the episode counter
            break

# Save the trained models
torch.save(policy_net1.state_dict(), f"{env.game_modes.game_type}_agent1.pth")
torch.save(policy_net2.state_dict(), f"{env.game_modes.game_type}_agent2.pth")
print("Training complete, models saved.")