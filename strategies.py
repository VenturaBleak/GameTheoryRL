import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Strategy:
    def __init__(self, env):
        self.env = env

    def select_action(self, observation, model=None):
        raise NotImplementedError

    def get_last_opponent_action(self, observation):
        if self.env.current_round == 0:
            raise ValueError("Cannot get last opponent action when no history")
        last_opponent_action = observation[self.env.current_round * 2 - 1]
        return last_opponent_action

    def get_effective_opponent_actions(self, observation):
        mask_start = 2 * self.env.max_rounds
        # Pair each opponent action with its corresponding mask and filter
        return [observation[i] for i in range(1, mask_start, 2) if observation[mask_start + i // 2] == 1]

    def opponent_cooperation_ratio(self, observation):
        effective_actions = self.get_effective_opponent_actions(observation)
        # print for a sanity check
        # print(f"-------------------")
        # print(f"current_round: {self.env.current_round}")
        # print(f"observation: {observation}")
        # print(f"effective_actions: {effective_actions}")
        if not effective_actions:
            return 0.5  # Default value when no effective history
        return 1 - sum(effective_actions) / len(effective_actions)  # cooperation is represented by 0

class RandomPick(Strategy):
    def select_action(self, observation, model=None):
        return self.env.action_space.sample()


class Defector(Strategy):
    def select_action(self, observation, model=None):
        return 1


class Cooperator(Strategy):
    def select_action(self, observation, model=None):
        return 0


class ModelStrategy(Strategy):
    def select_action(self, observation, model=None):
        with torch.no_grad():
            state = torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(0)
            selected_action = model(state).argmax(dim=1).view(1, 1).squeeze(0).item()
            return selected_action


class TitForTat(Strategy):
    def select_action(self, observation, model=None):
        return 0 if self.env.current_round == 0 else self.get_last_opponent_action(observation)


class Alternator(Strategy):
    def select_action(self, observation, model=None):
        return 0 if self.env.current_round % 2 == 0 else 1


class Appeaser(Strategy):
    def select_action(self, observation, model=None):
        effective_actions = self.get_effective_opponent_actions(observation)
        last_opponent_action = effective_actions[-1] if effective_actions else None
        if last_opponent_action == 1:
            return 1 - observation[-3]  # previous own action
        return observation[-3]  # previous own action

class AverageCopier(Strategy):
    def select_action(self, observation, model=None):
        rnd_num = random.random()
        if rnd_num < self.opponent_cooperation_ratio(observation):
            return 0
        return 1

class Grudger(Strategy):
    def select_action(self, observation, model=None):
        effective_actions = self.get_effective_opponent_actions(observation)
        if 1 in effective_actions:
            return 1
        return 0

class GoByMajority(Strategy):
    def select_action(self, observation, model=None):
        effective_actions = self.get_effective_opponent_actions(observation)
        if sum(effective_actions) > len(effective_actions) / 2:
            return 0
        return 1

class StrategyFactory:
    strategy_probabilities = {
        "random": (0., RandomPick),
        "defector": (0., Defector),
        "cooperator": (0., Cooperator),
        "model": (0., ModelStrategy),
        "tit_for_tat": (1., TitForTat),
        "alternator": (0., Alternator),
        "appeaser": (0., Appeaser),
        "average_copier": (0., AverageCopier),
        "grudger": (0., Grudger),
        "go_by_majority": (0., GoByMajority),
    }

    @classmethod
    def create_strategy(cls, env, strategy_type="sample_from_dict"):
        if strategy_type == "sample_from_dict":
            strategy_type = cls.sample_strategy()

        assert strategy_type in cls.strategy_probabilities, \
            f"Strategy type must be one of {list(cls.strategy_probabilities.keys())} or 'sample_from_dict'"

        return cls.strategy_probabilities[strategy_type][1](env)

    @staticmethod
    def sample_strategy():
        strategies = list(StrategyFactory.strategy_probabilities.keys())
        weights = [prob[0] for prob in StrategyFactory.strategy_probabilities.values()]
        return random.choices(strategies, weights=weights)[0]