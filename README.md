# Reinforcement Learning: Game Theory with Prisoner's Dilemma

## Overview
The repository delves deep into the intricacies of the well-known problem in game theory - the Prisoner's Dilemma. In this classic problem, two players face a situation where they can either choose to cooperate and reap mutual benefits or betray each other, potentially gaining more but at the risk of mutual loss.

The Prisoner's Dilemma is defined by its characteristic payoff matrix: (3,3) if both players cooperate, (5,0) and (0,5) if one defects while the other cooperates, and (1,1) if both defect. Through the lens of reinforcement learning, specifically using Deep Q Networks (DQN), this repo captures the intricacies of agent interactions, illustrating how agents can be trained to either cooperate or defect based on different training regimes.

By leveraging DQN, agents learn to optimize their actions through interactions with the environment. DQN employs a neural network to approximate the Q-values (expected values of a certain state as the result of an action taken), which are updated using experience replay, ensuring a stable learning process.

### Hand-Coded Game Visualization in Pygame
The repository  features a hand-coded visualization of the Prisoner's Dilemma using the pygame framework. This visualization brings the game's dynamics to life, illustrating in real-time how the agents, trained using DQN, react and strategize.

![Prisoner's Dilemma](https://github.com/VenturaBleak/GameTheoryRL/blob/a66203a97bb44b1382cc66af64500fe22e0175a2/PrisonersDilemma.png)

### Strategies:
Various strategies are implemented to understand the dynamics of the game:

- Defector
- Cooperator
- Random
- Alternator
- Grudger
- TitForTat
- ... and more.

These strategies provide a spectrum of behaviors, from unwavering cooperation to deterministic betrayal, offering diverse training opponents for the DQN agent.

## Code Structure

### Files in the Repository:

**Scripts and Environments:**<br>
1. **buffer.py**: Manages the replay buffer, essential for stable DQN training.
2. **evaluate.py**: Evaluates the performance of trained agents.
3. **game_env.py**: The core game environment for Prisoner's Dilemma.
4. **game_evaluate.py**: A tool for evaluating various game dynamics.
5. **game_modes.py**: Houses different game modes for experimentation.
6. **game_renderer.py**: Visual renderer in pygame for an interactive experience.
7. **game_visualize.py**: Offers insights into the game dynamics through visualizations.
8. **hyperparameters.py**: Configuration file for various training parameters.
9. **strategies.py**: Implementations of classic and custom game strategies.
10. **tournament.py**: Hosts a tournament where every agent competes against every other.
11. **train.py**: Main training script for the DQN agent.
12. **train_with_curriculum.py**: Employs a curriculum-based approach for sequential learning.
13. **train_with_strategy.py**: Pits the DQN agent against specific strategies for targeted learning.

<br>

**Trained Models:**<br>
14. **prisoners_dilemma_agent1.pth**: Weights for the first agent after training.
15. **prisoners_dilemma_agent2.pth**: Weights for the second agent post-training.

### How to Run:

1. **Train the model**:
    ```bash
    python train.py --episodes 5000
    ```

2. **Train with a specific strategy**:  
    ```bash
    python train_with_strategy.py --strategy TitForTat
    ```

3. **Visualize game dynamics**:  
    ```bash
    python game_visualize.py
    ```

4. **Evaluate model's performance**:  
    ```bash
    python evaluate.py
    ```

5. **Run a tournament**:  
    ```bash
    python tournament.py
    ``
