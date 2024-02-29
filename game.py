import numpy as np
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from agent import Agent
import concurrent.futures


class MinorityGame:
    def __init__(self, num_agents, memory_size, num_rounds, explorationRounds, minority_threshold):
        self.num_agents = num_agents
        self.agents = [Agent(memory_size, explorationRounds=explorationRounds, minority_threshold=minority_threshold) for _ in range(num_agents)]
        self.num_rounds = num_rounds
        self.memory_size = memory_size
        self.history = np.random.choice([0, 1], size=memory_size).tolist()
        self.strategy_decision_counts = {strategy: [] for strategy in self.agents[0].strategy_types}
        self.strategy_counts_per_round = {strategy: [] for strategy in self.agents[0].strategy_types}
        self.minority_threshold = minority_threshold
        
    def play_round(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            decisions = list(executor.map(lambda agent: agent.decide(self.history), self.agents))
        decisionCount = {0:0,1:0}
        for decision in decisions:
            decisionCount[decision] += 1
        if decisionCount[0]/(self.num_agents) > self.minority_threshold:
            minority_decision = 1
        else:
            minority_decision = 0
        round_strategy_counts = {strategy: 0 for strategy in self.agents[0].strategy_types}
        strategy_decisions = {strategy: 0 for strategy in self.agents[0].strategy_types}
        strategy_counts = {strategy: 0 for strategy in self.agents[0].strategy_types}
        for agent, decision in zip(self.agents, decisions):
            outcome = 1 if decision == minority_decision else 0
            agent.update_score(outcome) 
            agent.update_strategy_data(self.history, decision, outcome)
            round_strategy_counts[agent.current_strategy] += 1
            strategy_decisions[agent.current_strategy] += decision == minority_decision
            strategy_counts[agent.current_strategy] += 1
        for strategy in self.agents[0].strategy_types:
            self.strategy_counts_per_round[strategy].append(round_strategy_counts[strategy])
            if strategy_counts[strategy] > 0:
                percentage = (strategy_decisions[strategy] / strategy_counts[strategy]) * 100
            else:
                percentage = 0
            self.strategy_decision_counts[strategy].append(percentage)
        self.history.pop(0)
        self.history.append(minority_decision)
        
    def simulate(self):
        for _ in range(self.num_rounds):
            self.play_round()