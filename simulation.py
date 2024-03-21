import numpy as np
import hashlib
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from agent import Agent
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
from game import MinorityGame
from scipy.stats import norm
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.collections import PolyCollection


class Simulation:
    def __init__(self, num_agents, memory_size, num_rounds, explorationRounds, game_count, minority_threshold, inductive):
        init_params = (num_agents, memory_size, num_rounds, explorationRounds, game_count, minority_threshold, inductive)
        self.file_name = f"saves/{hashlib.sha256(str(init_params).encode()).hexdigest()}.pkl"
        
        if os.path.exists(self.file_name):
            self.load()
            print("Loaded existing data")
        else:
            self.num_agents = num_agents
            self.memory_size = memory_size
            self.num_rounds = num_rounds
            self.explorationRounds = explorationRounds
            self.game_count = game_count
            self.all_games_strategy_counts = {strategy: [] for strategy in Agent(memory_size, explorationRounds, minority_threshold, inductive).strategy_types}
            self.all_games_rewards = []
            self.games=[]
            self.strategy_switches = {strategy: {s: 0 for s in Agent(memory_size, explorationRounds,minority_threshold, inductive).strategy_types} for strategy in Agent(memory_size, explorationRounds,minority_threshold, inductive).strategy_types}
            self.minority_threshold = minority_threshold
            self.inductive = inductive
            print("Simulating Minority Game...")
            self.run_simulation()  # Automatically run the simulation if data does not exist.
            self.save()  # Automatically save the data after simulation.

    def save(self):
        with open(self.file_name, 'wb') as file:
            pickle.dump(self, file)

    def load(self):
        with open(self.file_name, 'rb') as file:
            data = pickle.load(file)
            self.__dict__.update(data.__dict__)


    def run_simulation(self):
        for _ in range(self.game_count):
            #print the progress, overwriting the previous line
            print(f"Running game {_+1}/{self.game_count}\t\t", end="\r")
            game = MinorityGame(self.num_agents, self.memory_size, self.num_rounds, self.explorationRounds, self.minority_threshold, self.inductive)
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
        plt.figure(figsize=(6, 5))  # Adjusted figure size for better fit
        
        # Calculate total counts per round to normalize
        total_counts_per_round = np.zeros(self.num_rounds)
        for strategy, counts in self.all_games_strategy_counts.items():
            total_counts_per_round += np.array(counts)
        
        # Avoid division by zero in rounds where total_counts_per_round might be 0
        total_counts_per_round[total_counts_per_round == 0] = 1
        
        for strategy, counts in self.all_games_strategy_counts.items():
            # Normalize counts by total counts per round
            normalized_counts = np.array(counts) / total_counts_per_round
            plt.bar(rounds, normalized_counts, label=strategy, bottom=bottom)
            bottom += normalized_counts  # Update bottom for stack effect
        
        plt.xlabel('Round')
        plt.ylabel('Proportion of Agents')
        # plt.title('Average Distribution of Strategies Over Time Across All Games (Normalized)')
        plt.xticks(np.arange(0, self.num_rounds, 10), rotation=90)

        # Position the legend below the plot
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        plt.show()

    def plot_combined_rewards_distribution(self):
        plt.figure(figsize=(6, 4))
        maxScore = int(np.max(self.all_games_rewards))
        minScore = int(np.min(self.all_games_rewards))
        diff = int((maxScore - minScore) / 2)
        
        # Plot histogram with density=True to scale it as a probability distribution
        hist_data, bins, _ = plt.hist(self.all_games_rewards, bins=diff, alpha=0.7, density=True)
        
        # Calculate the mean and standard deviation for the normal distribution
        mu, sigma = np.mean(self.all_games_rewards), np.std(self.all_games_rewards)
        
        # Create a range of x values that covers the distribution
        x = np.linspace(minScore, maxScore, 1000)
        
        # Calculate the normal distribution with the same mean and standard deviation
        p = norm.pdf(x, mu, sigma)
        
        # Plot the normal distribution curve
        plt.plot(x, p, 'b', linewidth=2)

        # add text at the peak of the normal distribution, with the mean value
        plt.text(mu, np.max(p)/2, f'Mean\n {mu:.2f}', ha='center', va='bottom', color='b', size=14)
        
        plt.xlabel('Total Rewards')
        plt.ylabel('Probability')

        # plt.title('Normalized Distribution of Total Rewards Across All Agents in All Games')
        plt.show()

    def plot_switching_heatmap(self):
        # Convert switch data to a matrix for heatmap plotting
        strategies = list(self.strategy_switches.keys())
        switch_matrix = np.array([[self.strategy_switches[row][col] for col in strategies] for row in strategies])

        # Create a mask for the diagonal
        mask = np.eye(len(strategies), dtype=bool)

        plt.figure(figsize=(6, 5))
        # Pass the mask to the heatmap function
        sns.heatmap(switch_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=strategies, yticklabels=strategies, mask=mask)
        # plt.title('Strategy Switching Heatmap Across All Games')
        plt.xlabel('To Strategy')
        plt.ylabel('From Strategy')
        plt.show()

    def plot_average_strategy_success(self):
        # Calculate the average success rate for each strategy
        # plot as a bar chart, with one bar for each strategy
        strategy_success = {strategy: [] for strategy in Agent(self.memory_size, self.explorationRounds,self.minority_threshold, self.inductive).strategy_types}
        for game in self.games:
            for strategy, percentages in game.strategy_decision_counts.items():
                strategy_success[strategy].append(percentages)
        average_success = {strategy: np.mean(successes) for strategy, successes in strategy_success.items()}
        plt.figure(figsize=(6, 4))
        plt.bar(average_success.keys(), average_success.values())
        plt.xlabel('Strategy')
        plt.ylabel('Average Success Rate, %')
        plt.xticks(rotation=45)
        # plt.title('Average Success Rate for Each Strategy Across All Games')
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
        fig, ax = plt.subplots(figsize=(6, 4))
        agents_indices = np.arange(len(sorted_agent_info))
        bottom = np.zeros(len(sorted_agent_info))
        for strategy in self.games[0].agents[0].strategy_types:
            ax.bar(agents_indices, strategy_reward_counts[strategy], bottom=bottom, label=strategy)
            bottom += np.array(strategy_reward_counts[strategy])
        ax.set_xlabel('Agents (Sorted by Total Rewards)')
        ax.set_ylabel('Number of Rewards')
        # ax.set_title('Rewards per Strategy for Each Agent (Sorted by Total Rewards)')
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
        fig, ax = plt.subplots(figsize=(6, 4))
        agents_indices = np.arange(len(sorted_agent_info))
        bottom = np.zeros(len(sorted_agent_info))
        for strategy in self.games[0].agents[0].strategy_types:
            ax.bar(agents_indices, strategy_reward_counts[strategy][::-1], bottom=bottom, label=strategy)
            bottom += np.array(strategy_reward_counts[strategy])
        ax.set_xlabel('Agents (Sorted by Total Rewards)')
        ax.set_ylabel('Number of Rewards')
        # ax.set_title('Wins per Strategy for Each Agent (Sorted by Total Rewards)')
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
        fig, ax = plt.subplots(figsize=(6, 4))

        # Calculate and plot the number of agents selecting option 0 for each round for each game
        for game_idx, game in enumerate(self.games):
            option_0_counts = []
            for round_decision in zip(*[agent.choices for agent in game.agents]):
                option_0_count = round_decision.count(0)
                option_0_counts.append(option_0_count)
            all_games_option_0_counts[game_idx, :] = option_0_counts
            # convert to a percentage of the total number of agents
            option_0_counts = np.array(option_0_counts) / self.num_agents * 100
            plt.plot(rounds, option_0_counts, label=f'Game {game_idx + 1}', alpha=0.5, linewidth=1)
        
        # Calculate the average number of agents selecting option 0 across all games and plot it
        avg_option_0_counts = np.mean(all_games_option_0_counts, axis=0)
        plt.plot(rounds, avg_option_0_counts, label='Average', color='blue', linewidth=2, linestyle='-')
        
        #plot the minority threshold * number of agents as a horizontal line
        plt.axhline(self.minority_threshold * 100 , color='g', linestyle='--', label='Minority Threshold', linewidth=3)

        plt.xlabel('Round')
        plt.ylabel('Percentage of Agents Selecting Option 0')
        # plt.title('Percentage of Agents Selecting Option 0 per Round Across All Games')
        legend_elements = [Line2D([0], [0], color='r', lw=2, alpha=0.3, label='Games'),
                           Line2D([0], [0], color='g', lw=2, linestyle='--', label='Minority Threshold'),
                           Line2D([0], [0], color='blue', lw=2, label='Overall Average')]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_agents_in_minority_single_shot(self):
        rounds = np.arange(self.num_rounds)
        overall_option_0_counts = []

        for game in self.games:
            for round_idx in rounds:
                round_decision = [agent.choices[round_idx] for agent in game.agents]
                option_0_count = round_decision.count(0)
                percentage = (option_0_count / self.num_agents) * 100
                overall_option_0_counts.append({'Round': round_idx, 'Percentage': percentage})

        # Convert list of dicts into a DataFrame
        df = pd.DataFrame(overall_option_0_counts)

        # Plotting starts here
        plt.figure(figsize=(6,4))

        plt.axhline(self.minority_threshold * 100, color='c', linestyle='--', label='Minority Threshold', linewidth=3, zorder=-1)

        # Overlaying the box plot first with filled color
        sns.boxplot(x='Round', y='Percentage', data=df, width=0.1, showmeans=False,
                    boxprops={'facecolor': 'blue', 'edgecolor': 'black'},
                    whiskerprops={'color': 'black'},
                    capprops={'color': 'black'},
                    medianprops={'color': 'red'},
                    flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'black'})

        # Adding the violin plot
        sns.violinplot(x='Round', y='Percentage', data=df, inner=None, color="b")

        # Adjusting opacity of the violin plots
        for art in plt.gca().findobj(PolyCollection):
            art.set_alpha(0.5)

        # Plot the minority threshold line

        plt.ylabel('Percentage of Agents Selecting Option 0')
        #hide xticks
        plt.xticks([])
        #hide x-axis label
        plt.xlabel('')
        plt.grid(True)
        
        plt.legend( loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def mean_agent_success_rate_percentage(self):
        return round(100*(np.mean(self.all_games_rewards)/self.num_rounds),2)