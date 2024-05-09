from modules.dueling_deep_network import DuelingDeepNetwork
from modules.agent import Agent

from collections import Counter
from time import time
import numpy as np

import mo_gymnasium as mo_gym
import gymnasium as gym
import torch.nn as nn
import torch as T

from sys import argv

import warnings
warnings.filterwarnings('ignore')

R1, R2, R3, R4 = 80, 145, 166, 175

TREASURE_MAP = {
    50: 'Ø',
    R1 + 50: 'R1',
    R2 + 50: 'R2',
    R3 + 50: 'R3',
    R4 + 50: 'R4',

    (R1 + R2) + 50: 'R1-R2',
    (R1 + R3) + 50: 'R1-R3',
    (R1 + R4) + 50: 'R1-R4',
    (R2 + R3) + 50: 'R2-R3',
    (R2 + R4) + 50: 'R2-R4',
    (R3 + R4) + 50: 'R3-R4',

    (R1 + R2 + R3) + 50: 'R1-R2-R3',
    (R1 + R2 + R4) + 50: 'R1-R2-R4',
    (R1 + R3 + R4) + 50: 'R1-R3-R4',
    (R2 + R3 + R4) + 50: 'R2-R3-R4',

    (R1 + R2 + R3 + R4) + 50: 'R1-R2-R3-R4'
}

class MRG_DuelingDeepNetwork(DuelingDeepNetwork):
    def __init__(self, learning_rate: float, n_actions: int, input_dims: list, name: str, chkpt_dir: str) -> None:
        super(MRG_DuelingDeepNetwork, self).__init__(learning_rate, n_actions, input_dims, name, chkpt_dir)
        
        self.fc = nn.Sequential(
            nn.Linear(*input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

class MRG_Agent(Agent):
    def __init__(
        self, gamma: float, epsilon: float, learning_rate: float, n_actions: int, 
        input_dims: list, mem_size: int, batch_size: int, 
        eps_min: float = 0.01 , eps_decay: float = 5e-7, 
        replace: int = 1000, 
        chkpt_dir: str = 'backup', agent_name: str = 'mrg_agent'
    ) -> None:
        super(MRG_Agent, self).__init__(
            gamma, epsilon, learning_rate, n_actions, input_dims, mem_size, 
            batch_size, agent_name, eps_min, eps_decay, replace, chkpt_dir
        )

        self.q_eval = MRG_DuelingDeepNetwork(
            self.lr, self.n_actions, self.input_dims, 
            f'{agent_name}_q_eval', 
            self.chkpt_dir
        )

        self.q_next = MRG_DuelingDeepNetwork(
            self.lr, self.n_actions, self.input_dims, 
            f'{agent_name}_q_next', 
            self.chkpt_dir
        )

def increment_heatmap(heatmap: np.array, episode_path: list[tuple]) -> None:
    """
        Increments the heatmap of visited cells based on the given episode path.

        Args:
            heatmap (np.array): The heatmap to be incremented.
            episode_path (list[tuple]): The path taken during the episode.

        Returns:
            None, the heatmap is modified in place.
    """

    positions_count = Counter(episode_path)

    for (row, col), count in positions_count.items():
        heatmap[row][col] += count

def generate_path_matrix(episode_path: list[tuple]) -> np.array:
    """
        Generates a matrix representing the path taken during an episode.

        Args:
            episode_path (list[tuple]): The path taken during the episode.

        Returns:
            np.array: A matrix representing the path taken during the episode.
    """

    matrix = np.zeros((5, 5), dtype=int)
   
    positions_count = Counter(episode_path)

    for (row, col), count in positions_count.items():
        matrix[row][col] += count

    return matrix

def is_cardinal_sequence(episode_path: list[tuple]) -> bool:
    """
        Checks whether the episode path is a cardinal sequence, meaning that all 
        states in the sequence are in one of the four cardinal positions of the preceding state..

        Args:
            episode_path (list[tuple]): The path taken during the episode.

        Returns:
            bool: True if the episode path is a cardinal sequence, False otherwise.
    """

    if len(episode_path) < 2:
        return False
    
    # Cardinal steps
    cardinal_steps = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for idx in range(1, len(episode_path)):
        x_diff = episode_path[idx][0] - episode_path[idx - 1][0]
        y_diff = episode_path[idx][1] - episode_path[idx - 1][1]
        
        # Check if the difference between coordinates matches a cardinal step
        if (x_diff, y_diff) not in cardinal_steps:
            return False
    
    return True

def not_worse_path(path_len: int, treasure: int, converged_paths: dict) -> bool:
    """
        Checks whether the given path is not worse than the current best path to a given the treasure.

        Args:
            path_len (int): The length of the path to the treasure.
            treasure (int): The treasure to reach.
            converged_paths (dict): A dictionary containing the current best paths to the treasures.

        Returns:
            bool: True if the given path is not worse than the current best path to the treasure, False otherwise.
    """
    
    if treasure in [ solution[1] for solution in converged_paths.keys() ]:

        current_len_for_treasure = next(solution for solution in converged_paths.keys() if solution[1] == treasure)[0]

        if path_len > current_len_for_treasure: return False

    return True

def learn_env(
    env: gym.Env, agent: Agent, 
    conversion_threshold: int = 300,
    n_trial: int = 1,
    write_results: bool = False, 
) -> None:
    """
        Trains the agent in the given environment.

        Args:
            env (gym.Env): The environment to train the agent in.
            agent (Agent): The agent to be trained.
            conversion_threshold (int, optional): The threshold for considering a path as converged. Defaults to 300.
            n_trial (int, optional): The number of trials. Defaults to 1.
            write_results (bool, optional): Whether to write the results. Defaults to False.
    """

    n_episodes = 100000

    scores, eps_history = [], []
    loss_history, episode_losses = [], []
    actions_history = []

    # Saves the paths taken in the episodes to check for converged states
    paths_hashs = Counter()

    # Saves the paths of conversion  to an optimal solution { solution_final_reward: path }
    converged_paths = {}
    removed_converged_paths = {}

    # Saves the episode of conversion to an optimal solution: { episode: path_length }
    converged_episodes = {}

    # Prevents the agent from reaching solutions with the same evaluation
    converged_evals = []

    # Map space tresure locations
    treasure_locs = set([ (0, 2), (2, 2), (4, 2), (4, 0) ])

    # Expected solution set
    expected_solutions = set([
        (8, 130), (8, 195), (8, 216), # One treasure
        (8, 275), (8, 361), (8, 391), # Two treasures
        (8, 441),                     # Three treasures
        (12, 616)                     # Four treasures 
    ])

    # Matrix to save the position visitation
    heatmap = np.zeros(env.map.shape, dtype=int)

    gold_home = 0
    got_lost = 0

    forced_paths = 0
    tried_paths = 0

    boosted_paths = 0
    boosted_long_paths = 0

    last_epsilon_reset = 0
    eps_since_valid_path = 0

    for episode in range(n_episodes):

        # Checking if the agent converged all optimal solutions
        if len(expected_solutions & set(converged_paths.keys())) >= 7:
            print('Agent converged for desired solutions')
            break
        
        observation, _ = env.reset()  

        done = False
        score = 0
        treasure = 0
        
        episode_losses = [0]
        actions_type = []
        transitions = []
        
        episode_path = []
        episode_path_hash = None
        
        stop = False

        while not done:       
            action, action_type = agent.choose_action(observation)

            action_would_take_home = tuple(env.current_pos + env.dir[action]) == tuple(env.final_pos)
            found_equivalent_path  = score + 24.5 in converged_evals

            current_solution = next((solution for solution in converged_paths.keys() if solution[1] == treasure + 50), False)
            better_path = len(episode_path) + 1 < current_solution[0] if current_solution and action_would_take_home else False

            # If an improved solution is identified compared to the one currently learned, the existing solution is 
            # discarded, allowing the agent to pursue the superior alternative
            if better_path:
                converged_evals = [ eval for eval in converged_evals if eval != score + 24.5]

                removed_converged_paths[current_solution] = converged_paths[current_solution]
                del converged_paths[current_solution]

                old_conversion_ep = next((key for key, val in converged_episodes.items() if val == score + 24.5), None)
                del converged_episodes[old_conversion_ep]

                break

            # Stops the agent from going to the final state when an equivalent solution has already been found
            if action_would_take_home and found_equivalent_path and current_solution and not better_path:
                
                low_visits = np.argwhere(
                    (heatmap <= np.mean(heatmap)) &
                    ~generate_path_matrix([(0, 0), *episode_path]).astype(bool)
                )
                new_pos = low_visits[np.random.randint(0, len(low_visits))]

                env.set_state(new_pos)
                episode_path.append(tuple(new_pos))

                action, _ = agent.choose_action(new_pos)
                action_type = 'Forced'
                forced_paths += 1
       
            actions_type.append(action_type)

            next_observation, reward, done, _, info = env.step(action)

            episode_path.append(tuple(env.current_pos))

            score += reward
            treasure += info['vector_reward'][1]
            
            agent.store_transition(observation, action, reward, next_observation, int(done))
            transitions.append((observation, action, reward, next_observation, int(done)))
            
            loss = agent.learn()
            
            episode_losses.append(loss)

            observation = next_observation

            if len(episode_path) == 100:
                got_lost += 1
                eps_since_valid_path += 1
                break
        
        else:
            if is_cardinal_sequence(episode_path) and not_worse_path(len(episode_path), int(treasure), converged_paths):

                eps_since_valid_path = 0

                episode_path_hash = f'{hash(str(episode_path))}{len(episode_path):03d}{treasure:05.1f}'
                paths_hashs[episode_path_hash] += 1

                good_short_path = len(episode_path) == 8 and treasure_locs & set(episode_path)
                good_long_path  = 8 < len(episode_path) <= 14 and len(treasure_locs & set(episode_path)) == 4
                if (good_short_path or good_long_path) and len(converged_evals):

                    boosted_paths += 10
                    
                    for observation, action, reward, next_observation, done in [ transition for _ in range(10) for transition in transitions ]:
                        agent.store_transition(observation, action, reward + 10 if good_short_path else reward + 100, next_observation, done)
                    
                    for _ in range(10): agent.learn()

                    if 8 < len(episode_path) <= 14: boosted_long_paths += 1
                
        increment_heatmap(heatmap, episode_path)
        tried_paths += 1
        
        if stop: break
        
        if episode_path[-1] == (4, 4) and treasure_locs & set(episode_path):
            gold_home += 1

        scores.append(score)    
        eps_history.append(agent.epsilon)
        loss_history.append(np.nanmean(episode_losses))
        actions_history.append(dict(Counter(actions_type)))

        if episode - last_epsilon_reset > 1000 or eps_since_valid_path > 100:
            
            print(f'\nIncreasing epsilon\n')

            agent.eps_decay = 1e-3
            agent.epsilon = 0.7

            last_epsilon_reset = episode

            eps_history.append(agent.epsilon) 
        
        else:
            print(f'Episodes since last epsilon reset: {episode - last_epsilon_reset}\tEpisodes since last valid path: {eps_since_valid_path}')

        _, hash_count = paths_hashs.most_common(1)[0] if paths_hashs else (0, 0)
        if hash_count >= conversion_threshold:
            
            print('\n___________________')
            print('Converged to solution')
            print(f"{episode_path} {reward} {info['vector_reward']}")
            print('___________________\n\n')

            # Saving episode of conversion
            converged_episodes[episode] = score
            last_epsilon_reset = episode

            # Saving the paths of conversion
            converged_paths[(len(episode_path), int(treasure))] = episode_path

            # Blocks the agent from reaching solutions with the same evaluation
            converged_evals.append(score)

            # Increasing agent randomness
            agent.eps_decay = 1e-3
            agent.epsilon = 0.7   
            eps_history.append(agent.epsilon)        

            # Reseting paths taken in the episodes 
            agent.reset_memory()
            paths_hashs = Counter()
            gold_home = 0
            got_lost = 0
            tried_paths = 0
            forced_paths = 0
            boosted_paths = 0

            # Saves the agent's weights
            agent.save_best((len(episode_path), int(treasure)))

        print(f'\nEpisode {episode} of {n_episodes}')
        
        treasure_mapping = TREASURE_MAP[treasure] if episode_path[-1] == (4, 4) else TREASURE_MAP[treasure + 50]
        print(f'\tScore: {score:.2f}\tTreasure: {int(treasure)} ({treasure_mapping})\tAVG Score: {np.mean(scores[-100:]):.2f}\tMean Loss: {loss_history[-1]:3f}\tEpsilon: {eps_history[-1]:5f}')

        top_repetitions = dict(sorted(
            Counter([ 
                (int(path_hash[-8:-5]), float(path_hash[-5:])) for path_hash, reps in paths_hashs.most_common(5) for _ in range(reps)
            ]).items()
        ))
        print(f'\n\tUnique valid paths: {len(paths_hashs)} -> Top 5 repetitions: {top_repetitions}\n')

        gold_home_percent = f'{gold_home     / tried_paths * 100:.1f}' if tried_paths > 0 else '-%'
        got_lost_percent  = f'{got_lost      / tried_paths * 100:.1f}' if tried_paths > 0 else '-%'
        forced_percent    = f'{forced_paths  / tried_paths * 100:.1f}' if tried_paths > 0 else '-%'
        boosted_percent   = f'{boosted_paths / tried_paths * 100:.1f}' if tried_paths > 0 else '-%'
        print(f'\tReturned home with gold: {gold_home} ({gold_home_percent}%)\tGot lost: {got_lost} ({got_lost_percent}%)\tPaths with forced actions: {forced_paths} ({forced_percent}%)\tBoosted Paths: {boosted_paths} ({boosted_percent}%)')
        
        print(f'\tLatest episode path length: {len(episode_path)} {episode_path if len(episode_path) < 12 else f"[{episode_path[0]}, ..., {episode_path[-1]}]"}')

        print('\n\tActions taken in episode: NN: {NN}, Rand: {Rand}, Forced: {Forced}'.format_map(Counter(actions_type)))
        
        print(f'\n\tConverged epsiodes: {converged_episodes}')

        print(f'\n\tConverged solutions: {list(converged_paths.keys())}')

        print(f'\n\tBoosted long path: {boosted_long_paths}')

        print('\n')
        print(generate_path_matrix(episode_path))
        print(heatmap)
                    
        if write_results:

            with open(f'solutions/solution_rg_{conversion_threshold}_{n_trial}.txt', 'a') as solution_file:
                
                discovered_front = ' '.join([ 
                    f'{-1 * path_len} {treasure}' if path else '0 0'
                    for (path_len, treasure), path in sorted({
                        **converged_paths,
                        **{ key: [] for key in expected_solutions.difference(set(converged_paths.keys())) }
                    }.items())
                ])

                solution_file.write(f'{discovered_front}\n')

    print(f'Done with {episode}/{n_episodes}')
    print(f'\n\tConverged epsiodes: {converged_episodes}')
    print(f'\n\tConverged paths: {converged_paths}')

if __name__ == '__main__':

    if len(argv) < 3:
        print('\nUsage: python mrg_multiple_trials.py n_runs threshold_1 threshold_2 ... threshold_n\n')
        raise Exception('Missing arguments!')

    n_runs = int(argv[1])
    thresholds = [int(arg) for arg in argv[2:]]

    rg_times = {
        threshold: []
        for threshold in thresholds
    }

    print(f'Runing {n_runs} trials of learn_env for thresholds {thresholds}')

    for threshold in thresholds:

        for trial in range(1, 20):
            env = mo_gym.make('modified-resource-gathering-v0', render_mode='rgb_array')
            env = mo_gym.LinearReward(env, weight=np.array([0.5, 0.5]))

            agent = MRG_Agent(
                gamma=0.8,
                epsilon=1.0, eps_decay=3e-3,
                learning_rate=1e-4,
                n_actions=4,
                input_dims=[2],
                mem_size=10000,
                batch_size=10,
                replace=500,
                chkpt_dir='backups'
            )

            start_time = time()

            learn_env(env, agent, threshold, trial, True)

            end_time = time()

            rg_times[threshold].append(end_time - start_time)

            with open(f'solutions/mrg_times_{threshold}.txt', 'a') as times_file:
                times_file.write(f'{trial}\t{rg_times}\n')
