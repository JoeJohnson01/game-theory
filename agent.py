import numpy as np
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class Agent:
    def __init__(self, memory_size, explorationRounds, minority_threshold):
        self.explorationRounds = explorationRounds
        self.memory_size = memory_size
        self.strategy_types = ['random', 'weighted_random', 'genetic', 'bayesian', 'adaptive', 'market_based', 'pattern_recognition','repeat_last','inverse_last']
        self.current_strategy = np.random.choice(self.strategy_types)
        self.strategy_scores = {strategy: 0 for strategy in self.strategy_types}
        self.pattern_memory = []
        self.bayesian_params = [[1, 1] for _ in range(2 ** memory_size)]
        self.adaptive_memory = []
        self.model_trained = False
        self.history_index_cache = {}
        self.decisionHistory = []
        self.learning_rate = 0.1  
        self.decision_prices = [1, 1]
        self.genetic_strategies = np.random.randint(2, size=(10, 2 ** memory_size))
        self.genetic_performance = np.zeros(10)  
        self.temperature = 1 
        self.last_strategy_change = 0  
        self.strategy_failure_count = {strategy: 0 for strategy in self.strategy_types}
        self.strategy_rewards = {strategy: 0 for strategy in self.strategy_types}
        self.particle_position = np.random.rand(2 ** memory_size) 
        self.particle_velocity = np.random.rand(2 ** memory_size) 
        self.personal_best_position = self.particle_position.copy()
        self.minority_threshold = minority_threshold
        self.wonLastRound = None
        self.choices=[]


    def decide(self, history):
        self.select_strategy()
        self.decisionHistory.append(self.current_strategy)
        if self.current_strategy == 'random':
            choice = self._random_decide()
        elif self.current_strategy == 'weighted_random':
            choice = self._weighted_random_decide()
        elif self.current_strategy == 'genetic':
            choice = self._genetic_decide(history)
        elif self.current_strategy == 'bayesian':
            choice = self._bayesian_decide(history)
        elif self.current_strategy == 'adaptive':
            choice = self.adaptive_decide(history)
        elif self.current_strategy == 'market_based':
            choice = self._market_based_decide()
        elif self.current_strategy == 'pattern_recognition':
            choice = self._pattern_recognition_decide(history)
        elif self.current_strategy == 'repeat_last':
            choice = self._repeat_last_decide()
        elif self.current_strategy == 'inverse_last':
            choice = self._inverse_last_decide()
        else:
            raise ValueError(f'Invalid strategy: {self.current_strategy}')
        self.choices.append(choice)
        return choice

    def _weighted_random_decide(self):
        return int(np.random.rand() < self.minority_threshold)

    def _repeat_last_decide(self):
        if self.wonLastRound == None:
            return np.random.choice([0, 1])
        elif self.wonLastRound:
            return self.choices[-1]
        else:
            return 1 - self.choices[-1]


    def _inverse_last_decide(self):
        if self.wonLastRound == None:
            return np.random.choice([0, 1])
        elif self.wonLastRound:
            return 1 - self.choices[-1]
        else:
            return self.choices[-1]

    def _random_decide(self):
        return np.random.choice([0, 1])
        
    def _genetic_decide(self, history):
        history_index = self._get_history_index(history)
        best_strategy_index = np.argmax(self.strategy_scores['genetic'])
        return self.genetic_strategies[best_strategy_index][history_index]
    
    def _bayesian_decide(self, history):
        history_index = self._get_history_index(history)
        alpha, beta = self.bayesian_params[history_index]
        return 0 if np.random.rand() < alpha / (alpha + beta) else 1

    def _pattern_recognition_decide(self, history):
        history_str = ''.join(map(str, history))
        decision = np.random.choice([0, 1])
        pattern_found = False
        next_move = None
        for past_history in self.pattern_memory:
            if len(past_history) > len(history_str) and past_history.endswith(history_str):
                pattern_found = True
                next_move = past_history[-len(history_str)-1]
                break
        if pattern_found and next_move:
            decision = 0 if next_move == '1' else 1  
        if history_str not in self.pattern_memory:
            self.pattern_memory.append(history_str)
        return decision


    def _market_based_decide(self):
        decision = 0 if self.decision_prices[0] < self.decision_prices[1] else 1
        if self.decision_prices[0] == self.decision_prices[1]:
            decision = np.random.choice([0, 1])
        return decision

    def adaptive_decide(self, history):
        if not self.adaptive_memory or np.random.rand() < self.learning_rate:
            return np.random.choice([0, 1])
        else:
            recent_decisions = self.adaptive_memory[-self.memory_size:]
            predicted_majority = int(np.mean(recent_decisions) >= self.minority_threshold)
            return 1 - predicted_majority

    def update_score(self, reward):
        if reward == 0:
            self.strategy_scores[self.current_strategy] -= 1
            self.strategy_failure_count[self.current_strategy] += 1
            self.wonLastRound = False
        else:
            self.strategy_scores[self.current_strategy] += reward
            self.strategy_rewards[self.current_strategy] += 1  
            self.strategy_failure_count[self.current_strategy] = 0
            self.wonLastRound = True


    def select_strategy(self):
        if len(self.decisionHistory) < self.explorationRounds or self.strategy_failure_count[self.current_strategy] > 3:
            self.current_strategy = np.random.choice(self.strategy_types)
            self.last_strategy_change = len(self.decisionHistory)
        else:
            self.current_strategy = self.softmax_selection()


    def softmax_selection(self):
        scores = np.array([self.strategy_scores[strategy] for strategy in self.strategy_types])
        failure_adjustment = np.array([1 / (1 + self.strategy_failure_count[strategy]) for strategy in self.strategy_types])
        time_since_change = np.array([len(self.decisionHistory) - self.last_strategy_change if strategy == self.current_strategy else 0 for strategy in self.strategy_types])
        adjusted_scores = scores * failure_adjustment + time_since_change
        exp_scores = np.exp(adjusted_scores / self.temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        selected_strategy = np.random.choice(self.strategy_types, p=probabilities)
        if selected_strategy != self.current_strategy:
            self.last_strategy_change = len(self.decisionHistory)
        return selected_strategy

    def update_genetic_strategies(self, history, decision, outcome):
        history_index = self._get_history_index(history)
        for i, strategy in enumerate(self.genetic_strategies):
            if strategy[history_index] == decision:
                self.genetic_performance[i] += 1 if decision == outcome else -1
        if len(self.decisionHistory) % 10 == 0:
            self.evolve_genetic_strategies()

    def evolve_genetic_strategies(self):
        sorted_indices = np.argsort(self.genetic_performance)[::-1]
        num_parents = len(self.genetic_strategies) // 2
        parents = self.genetic_strategies[sorted_indices[:num_parents]]
        offspring = []
        for _ in range(len(self.genetic_strategies) - num_parents):
            # Choose parent indices from a 1-dimensional array of available parent indices
            parent_indices = np.random.choice(np.arange(num_parents), 2, replace=False)
            parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]
            crossover_point = np.random.randint(1, parent1.shape[0])
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring.append(child)
        mutation_rate = 0.01
        for child in offspring:
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(len(child))
                child[mutation_point] = 1 - child[mutation_point]
        self.genetic_strategies[sorted_indices[num_parents:]] = np.array(offspring)
        self.genetic_performance[sorted_indices[num_parents:]] = 0


    def update_strategy_data(self, history, decision, outcome):
        index = self._get_history_index(history)
        for i in range(len(self.bayesian_params)):
            if decision == outcome:
                self.bayesian_params[i][0] += 1
            else:
                self.bayesian_params[i][1] += 1
        self.adaptive_memory.append(decision)
        if len(self.adaptive_memory) > self.memory_size:
            self.adaptive_memory.pop(0)
        self.decision_prices[decision] += 0.1
        self.decision_prices[1 - decision] -= 0.1
        self.decision_prices = [max(1, price) for price in self.decision_prices]
        history_str = ''.join(map(str, history))
        if history_str not in self.pattern_memory:
            self.pattern_memory.append(history_str)
        self.update_genetic_strategies(history, decision, outcome)

    def _get_history_index(self, history):
        history_key = tuple(history)
        if history_key not in self.history_index_cache:
            self.history_index_cache[history_key] = int(''.join(map(str, history)), 2)
        return self.history_index_cache[history_key]