import numpy as np
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from agent import Agent
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
from game import MinorityGame

class Simulation:
    def __init__(self, num_agents, memory_size, num_rounds, explorationRounds, game_count, minority_threshold=0.5):
        self.num_agents = num_agents
        self.memory_size = memory_size
        self.num_rounds = num_rounds
        self.explorationRounds = explorationRounds
        self.game_count = game_count
        self.all_games_strategy_counts = {strategy: [] for strategy in Agent(memory_size, explorationRounds, minority_threshold=minority_threshold).strategy_types}
        self.all_games_rewards = []
        self.games=[]
        self.strategy_switches = {strategy: {s: 0 for s in Agent(memory_size, explorationRounds,minority_threshold=minority_threshold).strategy_types} for strategy in Agent(memory_size, explorationRounds,minority_threshold=minority_threshold).strategy_types}
        self.minority_threshold = minority_threshold

    

    def run_simulation(self):
        for _ in range(self.game_count):
            #print the progress, overwriting the previous line
            print(f"Running game {_+1}/{self.game_count}\t\t", end="\r")
            game = MinorityGame(self.num_agents, self.memory_size, self.num_rounds, self.explorationRounds, self.minority_threshold)
            game.simulate()
            self.games.append(game)
            # Aggregate strategy counts
            for strategy in game.strategy_counts_per_round.keys():
                if len(self.all_games_strategy_counts[strategy]) < len(game.strategy_counts_per_round[strategy]):
                    self.all_games_strategy_counts[strategy] = game.strategy_counts_per_round[strategy]
                else:
                    self.all_games_strategy_counts[strategy] = [sum(x) for x in zip(self.all_games_strategy_counts[strategy], game.strategy_counts_per_round[strategy])]
            self.all_games_rewards.extend([sum(agent.strategy_rewards.values()) for agent in game.agents])
            # Aggregate switches
            for agent in game.agents:
                for i in range(1, len(agent.decisionHistory)):
                    prev_strategy = agent.decisionHistory[i-1]
                    current_strategy = agent.decisionHistory[i]
                    if prev_strategy != current_strategy:
                        self.strategy_switches[prev_strategy][current_strategy] += 1
        # Average the strategy counts over all games
        for strategy in self.all_games_strategy_counts.keys():
            self.all_games_strategy_counts[strategy] = [x / self.game_count for x in self.all_games_strategy_counts[strategy]]

    def plot_combined_strategy_distribution(self):
        rounds = np.arange(self.num_rounds)
        bottom = np.zeros(self.num_rounds)
        plt.figure(figsize=(20, 10))
        for strategy, counts in self.all_games_strategy_counts.items():
            plt.bar(rounds, counts, label=strategy, bottom=bottom)
            bottom += np.array(counts)
        plt.xlabel('Round')
        plt.ylabel('Average Number of Agents')
        plt.title('Average Distribution of Strategies Over Time Across All Games')
        plt.xticks(rounds, rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_combined_rewards_distribution(self):
        plt.figure(figsize=(20, 10))
        plt.hist(self.all_games_rewards, bins=20, alpha=0.7)
        plt.xlabel('Total Rewards')
        plt.ylabel('Number of Agents')
        plt.title('Distribution of Total Rewards Across All Agents in All Games')
        plt.show()

    def plot_switching_heatmap(self):
        # Convert switch data to a matrix for heatmap plotting
        strategies = list(self.strategy_switches.keys())
        switch_matrix = np.array([[self.strategy_switches[row][col] for col in strategies] for row in strategies])

        # Create a mask for the diagonal
        mask = np.eye(len(strategies), dtype=bool)

        plt.figure(figsize=(10, 8))
        # Pass the mask to the heatmap function
        sns.heatmap(switch_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=strategies, yticklabels=strategies, mask=mask)
        plt.title('Strategy Switching Heatmap Across All Games')
        plt.xlabel('To Strategy')
        plt.ylabel('From Strategy')
        plt.show()

    def plot_average_strategy_success(self):
        # Calculate the average success rate for each strategy
        # plot as a bar chart, with one bar for each strategy
        strategy_success = {strategy: [] for strategy in Agent(self.memory_size, self.explorationRounds,minority_threshold=self.minority_threshold).strategy_types}
        for game in self.games:
            for strategy, percentages in game.strategy_decision_counts.items():
                strategy_success[strategy].append(percentages)
        average_success = {strategy: np.mean(successes) for strategy, successes in strategy_success.items()}
        plt.figure(figsize=(20, 10))
        plt.bar(average_success.keys(), average_success.values())
        plt.xlabel('Strategy')
        plt.ylabel('Average Success Rate')
        plt.title('Average Success Rate for Each Strategy Across All Games')
        plt.show()

    def plot_agent_scores_by_strategy(self):
        # Calculate total rewards for each agent in each game
        total_rewards_per_agent = [(sum(agent.strategy_rewards.values()), game_idx, agent_idx) 
                                    for game_idx, game in enumerate(self.games) 
                                    for agent_idx, agent in enumerate(game.agents)]
        
        # Sort the total rewards, along with game and agent indices
        sorted_agent_info = sorted(total_rewards_per_agent, key=lambda x: x[0])
        
        # Prepare data structures for plotting
        strategy_reward_counts = {strategy: [] for strategy in self.games[0].agents[0].strategy_types}
        for _, game_idx, agent_idx in sorted_agent_info:
            agent = self.games[game_idx].agents[agent_idx]
            for strategy in agent.strategy_types:
                strategy_reward_counts[strategy].append(agent.strategy_rewards[strategy])
        
        # plot the data as a dot plot, with agents on the x-axis and rewards on the y-axis. Plot each strategy as a separate series
        fig, ax = plt.subplots(figsize=(20, 10))
        agents_indices = np.arange(len(sorted_agent_info))
        bottom = np.zeros(len(sorted_agent_info))
        for strategy in self.games[0].agents[0].strategy_types:
            ax.bar(agents_indices, strategy_reward_counts[strategy], bottom=bottom, label=strategy)
            bottom += np.array(strategy_reward_counts[strategy])
        ax.set_xlabel('Agents (Sorted by Total Rewards)')
        ax.set_ylabel('Number of Rewards')
        ax.set_title('Rewards per Strategy for Each Agent (Sorted by Total Rewards)')
        #hide the x-axis labels, as there are too many to display
        plt.xticks([])
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_agent_losses_by_strategy(self):
        #plot agent losses by strategy, as a bar chart with one bar for each agent
        #sort by agent overall scores (total rewards) with the lowest scoring agents on the left
        #this is the opposite of the plot_agent_scores_by_strategy function (e.g. the agents are in the same order, but the bars are flipped)
        total_rewards_per_agent = [(sum(agent.strategy_rewards.values()), game_idx, agent_idx)
                                    for game_idx, game in enumerate(self.games)
                                    for agent_idx, agent in enumerate(game.agents)]
        sorted_agent_info = sorted(total_rewards_per_agent, key=lambda x: x[0])
        strategy_reward_counts = {strategy: [] for strategy in self.games[0].agents[0].strategy_types}
        for _, game_idx, agent_idx in sorted_agent_info:
            agent = self.games[game_idx].agents[agent_idx]
            for strategy in agent.strategy_types:
                strategy_reward_counts[strategy].append(agent.strategy_rewards[strategy])
        fig, ax = plt.subplots(figsize=(20, 10))
        agents_indices = np.arange(len(sorted_agent_info))
        bottom = np.zeros(len(sorted_agent_info))
        for strategy in self.games[0].agents[0].strategy_types:
            ax.bar(agents_indices, strategy_reward_counts[strategy][::-1], bottom=bottom, label=strategy)
            bottom += np.array(strategy_reward_counts[strategy])
        ax.set_xlabel('Agents (Sorted by Total Rewards)')
        ax.set_ylabel('Number of Rewards')
        ax.set_title('Wins per Strategy for Each Agent (Sorted by Total Rewards)')
        plt.xticks([])
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_agents_in_minority(self):
        #plot the number of agents in the minority for each round for each game, as a percentage of the total number of agents in the game
        #plot as a line chart, with one line for each game
        #this will show how often the minority was won by the agents
        #also plot the average number of agents in the minority across all games, and the minority threshold
        rounds = np.arange(self.num_rounds)
        all_games_option_0_counts = np.zeros((self.game_count, self.num_rounds))
        fig, ax = plt.subplots(figsize=(20, 10))
        # Calculate and plot the number of agents selecting option 0 for each round for each game
        for game_idx, game in enumerate(self.games):
            option_0_counts = []
            for round_decision in zip(*[agent.choices for agent in game.agents]):
                option_0_count = round_decision.count(0)
                option_0_counts.append(option_0_count)
            all_games_option_0_counts[game_idx, :] = option_0_counts
            plt.plot(rounds, option_0_counts, label=f'Game {game_idx + 1}', alpha=0.5, linewidth=1)
        
        # Calculate the average number of agents selecting option 0 across all games and plot it
        avg_option_0_counts = np.mean(all_games_option_0_counts, axis=0)
        plt.plot(rounds, avg_option_0_counts, label='Average', color='black', linewidth=2, linestyle='--')
        
        #plot the minority threshold * number of agents as a horizontal line
        plt.axhline(self.minority_threshold * self.num_agents, color='g', linestyle='--', label='Minority Threshold', linewidth=3)

        plt.xlabel('Round')
        plt.ylabel('Number of Agents Selecting Option 0')
        plt.title('Number of Agents Selecting Option 0 per Round Across All Games')
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()