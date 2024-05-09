import numpy as np

class ReplayBuffer:
    """
        A replay buffer class for storing and sampling transitions for reinforcement learning.
    """

    def __init__(self, max_size: int, input_shape: list) -> None:
        """
            Initializes the ReplayBuffer class.

            Parameters:
                - max_size (int): The maximum size of the replay buffer.
                - input_shape (list): The shape of the input state.

            Returns:
                - None
        """ 
        
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros(
            (self.mem_size, *input_shape),
            dtype=np.float32
        )

        self.new_state_memory = np.zeros(
            (self.mem_size, *input_shape),
            dtype=np.float32
        )

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        # Mask to discount potential features rewards that may come after the current state
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state: np.array, action: int, reward: float, state_: np.array, done: bool) -> None:
        """
            Stores a transition in the replay memory.

            Parameters:
                - state (np.array): The current state of the environment.
                - action (int): The action taken in the current state.
                - reward (float): The reward received for taking the action.
                - state_ (np.array): The next state of the environment.
                - done (bool): Indicates whether the episode is done after taking the action.

            Returns:
                - None
        """

        # Index of first free memory
        index = self.mem_cntr % self.mem_size

        # Stores the transition on the memories in the indices in the appropriate arrays
        self.state_memory[index] = state
        self.new_state_memory[index] = state_

        self.action_memory[index] = action
        self.reward_memory[index] = reward

        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int) -> tuple:
        """
            Randomly samples a batch of transitions from the replay memory buffer.

            Args:
                batch_size (int): The number of transitions to sample.

            Returns:
                tuple: A tuple containing the sampled states, actions, rewards, next states, and terminal flags.
        """

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
    