from modules.agent import Agent

from collections import Counter
from time import time
import numpy as np

import mo_gymnasium as mo_gym
import gymnasium as gym
import sys

import threading
import warnings

warnings.filterwarnings('ignore')


def check_for_unblock(
        env: gym.Env, action: int, current_path: list,
        converged_paths: dict[tuple, list],
        converged_episodes: dict[tuple, int]
) -> tuple[dict, dict]:
    """
        Checks if a given action unblocks a path in the environment.

        Args:
            env (gym.Env): The environment object.
            action (int): The action to be taken.
            current_path (list): The current path taken.
            converged_paths (dict[tuple, list]): Dictionary of converged paths.
            converged_episodes (dict[tuple, int]): Dictionary of converged episodes.

        Returns:
            tuple[dict, dict]: A tuple containing the updated converged paths and converged episodes.
    """

    next_state = tuple(env.current_state + env.dir[action])
    converged_path = converged_paths.get(next_state, False)

    # The +1 represents the additional step that must be done if the current action is carried out.
    if converged_path and len(converged_path) > len(current_path) + 1:
        env.sea_map[next_state[0], next_state[1]] = env.treasures[next_state]

        new_converged_paths = {key: value for key, value in converged_paths.items() if key != next_state}
        new_converged_episodes = {key: value for key, value in converged_episodes.items() if key != next_state}

        return new_converged_paths, new_converged_episodes

    return converged_paths, converged_episodes


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
    del positions_count[(0, 0)]

    for (row, col), count in positions_count.items():
        heatmap[row][col] += count


def learn_env(
        env: gym.Env, agent: Agent,
        conversion_threshold: int = 300,
        n_trial: int = 1,
        load_checkpoint: bool = False,
        write_results: bool = False,
        save_backups: bool = False
) -> tuple:
    """
        Trains an agent to learn and navigate an environment.

        Args:
            env (gym.Env): The environment to train the agent on.
            agent (Agent): The agent to be trained.
            conversion_threshold (int, optional): The threshold for considering a state as converged. Defaults to 300.
            n_trial (int, optional): The trial number. Defaults to 1.
            load_checkpoint (bool, optional): Whether to load a checkpoint of the agent's models. Defaults to False.
            write_results (bool, optional): Whether to write the results to files. Defaults to False.
            save_backups (bool, optional): Whether to save the neural networks backups.

        Returns:
            tuple: A tuple containing the scores, epsilon history, loss history, actions history, heatmap, and converged episodes.
    """

    n_episodes = 100000

    if load_checkpoint:
        agent.load_models()

    scores, eps_history = [], []
    loss_history, episode_losses = [], []
    actions_history = []

    # Saves the paths taken in the episodes to check for converged states
    paths_hashs = Counter()

    # Saves the paths of conversion to a final state: { (i_final, j_final): path }
    converged_paths = {final_state: [] for final_state in env.treasures.keys()}

    # Saves the episode of conversion to a final state: { (i_final, j_final): episode }
    converged_episodes = {}

    # Matrix to save the position visitation
    heatmap = np.zeros(env.sea_map.shape)

    for episode in range(n_episodes):

        # Checking if the agent converged for treasures states
        if np.all(env.sea_map <= 0):
            print('Agent converged for all treasure states')
            break

        observation, _ = env.reset()
        done = False
        score = 0

        episode_losses = [0]
        actions_type = []

        episode_path = []
        episode_hash = None

        while not done:
            action, action_type = agent.choose_action(observation)
            actions_type.append(action_type)

            converged_paths, converged_episodes = check_for_unblock(
                env, int(action),
                episode_path,
                converged_paths, converged_episodes
            )

            next_observation, reward, done, _, _ = env.step(action)

            episode_path.append(tuple(env.current_state))

            score += reward

            agent.store_transition(observation, action, reward, next_observation, done)
            loss = agent.learn()

            episode_losses.append(loss)

            observation = next_observation

            if len(actions_type) == 1000:
                done = True

        else:
            increment_heatmap(heatmap, episode_path)

            episode_hash = hash(str(episode_path))
            paths_hashs[episode_hash] += 1

        scores.append(score)
        eps_history.append(agent.epsilon)
        loss_history.append(np.nanmean(episode_losses))
        actions_history.append(dict(Counter(actions_type)))

        _, hash_count = paths_hashs.most_common(1)[0]
        if hash_count >= conversion_threshold:

            converged_state = episode_path[-1]

            # Saving episode of conversion
            converged_episodes[converged_state] = episode

            # Saving the paths of conversion
            converged_paths[converged_state] = episode_path

            # Blocking converged state
            env.sea_map[converged_state[0], converged_state[1]] = -10

            # Increasing agent randomness
            agent.epsilon = 0.3
            agent.eps_decay = 1e-3

            # Resetting paths taken in the episodes 
            paths_hashs = Counter()

            if save_backups:
                agent.save_best(converged_state)

            env.reset()

            print(
                f'\n{env.name}\tRun {n_trial:03d} Converged to {len(converged_episodes):02d}/{len(converged_paths)} states')

        # print(f'Latest episode length: {len(episode_path)}')
        # print(f'Paths hashs: {len(paths_hashs)}\n{paths_hashs}\n')

        # print(f'Best path lengths: {env.best_paths_lengths}')
        # print(f'Converged lengths: { { final_state: len(path) for final_state, path in sorted(converged_paths.items()) } }')
        # print(f'Converged paths: {converged_paths}')
        # print(f'Converged episodes: {converged_episodes}')
        # print(f'Episode {episode} of {n_episodes}\n\tScore: {score:.2f} AVG Score: {np.mean(scores[-50:]):.2f} Mean Loss: {loss_history[-1]:3f} Epsilon: {agent.epsilon:5f}')
        # print('\tActions taken in episode: NN: {NN}, Rand: {Rand}'.format_map(Counter(actions_type)))
        # print(f'\tFinal state: {tuple(observation)}')

        if write_results:
            env_name = env.name.lower()
            with open(f'solutions/solution_{env_name}_{conversion_threshold}_{n_trial}.txt', 'a') as solution_file:
                discovered_front = ' '.join([
                    f'{-1 * len(path)} {int(env.treasures[final_state])}' if len(path) else '0 0'
                    for final_state, path in sorted(converged_paths.items())
                ])

                solution_file.write(f'{discovered_front}\n')

    return scores, eps_history, loss_history, actions_history, heatmap, converged_episodes


def run_dst(n_runs: int, thresholds: list[int]) -> None:
    print('Processing DST...')

    dst_times = {
        threshold: []
        for threshold in thresholds
    }

    for threshold in thresholds:

        for trial in range(n_runs):
            dst_env = mo_gym.make('deep-sea-treasure-v1', render_mode='rgb_array')
            dst_env = mo_gym.LinearReward(dst_env, weight=np.array([0.5, 0.5]))

            agent = Agent(
                gamma=0.99,
                epsilon=0.9, eps_decay=3e-3,
                learning_rate=1e-4,
                n_actions=4,
                input_dims=[2],
                mem_size=10000,
                batch_size=128,
                replace=500,
                chkpt_dir='backups',
                agent_name=f'dst_{trial}_{threshold}'
            )

            start_time = time()

            learn_env(dst_env, agent, threshold, trial, False, True)

            end_time = time()

            dst_times[threshold].append(end_time - start_time)

            with open('solutions/dst_times.txt', 'a') as dst_file:
                dst_file.write(f'Times after threshold {threshold}\n')
                dst_file.write(f'{dst_times}\n')
                dst_file.write('________________\n')

    print(f'{dst_times}')


def run_bst(n_runs: int, thresholds: list[int]) -> None:
    print('Processing BST...')

    bst_times = {
        threshold: []
        for threshold in thresholds
    }

    for threshold in thresholds:

        for trial in range(n_runs):
            bst_env = mo_gym.make('bountiful-sea-treasure-v1', render_mode='rgb_array')
            bst_env = mo_gym.LinearReward(bst_env, weight=np.array([0.5, 0.5]))

            agent = Agent(
                gamma=0.99,
                epsilon=0.9, eps_decay=3e-3,
                learning_rate=1e-4,
                n_actions=4,
                input_dims=[2],
                mem_size=10000,
                batch_size=128,
                replace=500,
                chkpt_dir='backups',
                agent_name=f'bst_{trial}_{threshold}'
            )

            start_time = time()

            learn_env(bst_env, agent, threshold, trial, False, True)

            end_time = time()

            bst_times[threshold].append(end_time - start_time)

            with open('solutions/bst_times.txt', 'a') as bst_file:
                bst_file.write(f'Times after threshold {threshold}\n')
                bst_file.write(f'{bst_times}\n')
                bst_file.write('________________\n')

    print(f'{bst_times}')


def run_mbst(n_runs: int, thresholds: list[int]) -> None:
    print(f'Processing MBST...')

    mbst_times = {
        threshold: []
        for threshold in thresholds
    }

    for threshold in thresholds:

        for trial in range(n_runs):
            mbst_env = mo_gym.make('modified-bountiful-sea-treasure-v1', render_mode='rgb_array')
            mbst_env = mo_gym.LinearReward(mbst_env, weight=np.array([0.5, 0.5]))

            agent = Agent(
                gamma=0.99,
                epsilon=0.9, eps_decay=3e-3,
                learning_rate=1e-4,
                n_actions=4,
                input_dims=[2],
                mem_size=10000,
                batch_size=128,
                replace=500,
                chkpt_dir='backups',
                agent_name=f'mbst_{trial}_{threshold}'
            )

            start_time = time()

            learn_env(mbst_env, agent, threshold, trial, False, True)

            end_time = time()

            mbst_times[threshold].append(end_time - start_time)

            with open(f'solutions/mbst_times.txt', 'a') as mbst_file:
                mbst_file.write(f'Times after threshold {threshold}\n')
                mbst_file.write(f'{mbst_times}\n')
                mbst_file.write('________________\n')

    print(f'{mbst_times}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('\nUsage: python dst_multiple_trials.py n_runs threshold_1 threshold_2 ... threshold_n\n')
        raise Exception('Missing arguments!')

    n_runs = int(sys.argv[1])
    thresholds = [int(arg) for arg in sys.argv[2:]]

    print(f'Processing {n_runs} runs for each Deep Sea Treasure Env with the following thresholds: {thresholds}')

    # Spawn a thread for each run function
    threads = []
    for func_name in ['run_dst', 'run_bst', 'run_mbst']:
        thread = threading.Thread(target=globals()[func_name], args=(n_runs, thresholds,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print('\nAll threads have finished running')
