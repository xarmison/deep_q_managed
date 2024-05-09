from .dueling_deep_network import DuelingDeepNetwork
from .replay_buffer import ReplayBuffer

import numpy as np
import torch as T

class Agent:
    """
        The Agent class represents an agent that interacts with the environment and learns to make decisions.
    """

    def __init__(
        self, gamma: float, epsilon: float, learning_rate: float, n_actions: int, 
        input_dims: list, mem_size: int, batch_size: int, agent_name: str, 
        eps_min: float = 0.01 , eps_decay: float = 5e-7, 
        replace: int = 1000, 
        chkpt_dir: str = 'backup'
    ) -> None:
        """
            Initializes the Agent object.

            Args:
                - gamma (float): Discount factor for future rewards.
                - epsilon (float): Exploration rate, determines the probability of taking a random action.
                - learning_rate (float): Learning rate for the neural network optimizer.
                - n_actions (int): Number of possible actions in the environment.
                - input_dims (list): Dimensions of the input state.
                - mem_size (int): Size of the replay memory buffer.
                - batch_size (int): Number of samples to train on in each learning iteration.
                - eps_min (float, optional): Minimum value for epsilon. Defaults to 0.01.
                - eps_decay (float, optional): Decay rate for epsilon. Defaults to 5e-7.
                - replace (int, optional): Number of steps before updating the target network. Defaults to 1000.
                - chkpt_dir (str, optional): Directory to save checkpoints. Defaults to 'backup'.
        """
       
        self. epsilon = epsilon
        self.lr = learning_rate
        self.gamma = gamma

        self.input_dims = input_dims
        self.n_actions = n_actions

        self.batch_size = batch_size
        self.mem_size = mem_size

        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.replace_target_cnt = replace
        self.learn_step_cnt = 0

        self.chkpt_dir = chkpt_dir

        self.action_space = [ action for action in range(self.n_actions) ]
        self.memory = ReplayBuffer(self.mem_size, self.input_dims)

        self.q_eval = DuelingDeepNetwork(
            self.lr, self.n_actions, self.input_dims,
            f'{agent_name}_q_eval',
            self.chkpt_dir
        )

        self.q_next = DuelingDeepNetwork(
            self.lr, self.n_actions, self.input_dims,
            f'{agent_name}_q_next',
            self.chkpt_dir
        )

    def choose_action(self, observation: list) -> tuple[int, str]:
        """
            Choose an action based on the given observation.

            Parameters:
                observation (list): The current observation.

            Returns:
                tuple[int, str]: A tuple containing the chosen action and its type.
                The first element is the action (an integer), and the second element is the action type (a string).
        """

        if np.random.random() > self.epsilon:
            # NN action
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.q_eval.device)

            _, advantage = self.q_eval.forward(state)
            
            action = T.argmax(advantage).item()
            action_type = 'NN'

        else:
            # Random action
            action = np.random.choice(self.action_space)
            action_type = 'Rand'

        return action, action_type

    def store_transition(self, state: np.array, action: int, reward: float, state_, done: bool) -> None:
        """
            Stores a transition in the replay memory buffer.

            Parameters:
                - state (np.array): The current state of the environment.
                - action (int): The action taken in the current state.
                - reward (float): The reward received for taking the action.
                - state_ (np.array): The next state of the environment.
                - done (bool): Indicates whether the episode is done after taking the action.

            Returns:
                - None
        """

        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self) -> None:
        """
            Replaces the target network with the evaluation network.

            This method is called periodically to update the target network with the weights of the evaluation network.
            The target network is used to estimate the Q-values for the next state during the training process.

            Returns:
                None
        """

        if self.learn_step_cnt % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            self.learn_step_cnt = 0

    def decrement_epsilon(self) -> None:
        """
            Decrements the value of epsilon by eps_decay if epsilon is greater than eps_min.
            If epsilon is already less than or equal to eps_min, it is set to eps_min.

            Returns:
                None
        """

        self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.eps_min else self.eps_min

    def save_models(self) -> None:
        """
            Saves the models' checkpoints.

            Returns:
                None
        """

        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def save_best(self, final_state: tuple) -> None:
        """
            Saves the models' checkpoints with the best score for a given final state.

            Parameters:
                - final_state (tuple): The final state of the environment.

            Returns:
                None
        """

        self.q_eval.save_best(final_state)
        self.q_next.save_best(final_state)

    def load_models(self) -> None:
        """
            Loads the models' checkpoints.

            Returns:
                None
        """

        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def reset_memory(self) -> None:
        """
            Resets the replay memory buffer.

            Returns:
                None
        """

        self.memory = ReplayBuffer(self.mem_size, self.input_dims)

    def learn(self) -> float:
        """
            Performs the learning process by randomly sampling the memory buffer to retrieve a batch_size sequence of actions.
            It then applies the learning equations to update the network weights.

            Returns:
                float: The loss value after the learning process.
        """

        # Wait until there have been batch size memory episodes 
        if self.memory.mem_cntr < self.batch_size:
            return np.nan

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        states  = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(next_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        V_s_, A_s_ = self.q_next.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        # Value rewards for which the next state is terminal
        q_eval[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        
        self.q_eval.optimizer.step()
        self.learn_step_cnt += 1

        self.decrement_epsilon()

        return loss.item()