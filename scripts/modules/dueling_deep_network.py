import torch.optim as optim
import torch.nn as nn
import torch as T
import os

class DuelingDeepNetwork(nn.Module):
    """
        A class for a dueling deep neural network for reinforcement learning.
    """

    def __init__(self, learning_rate: float, n_actions: int, input_dims: list, name: str, chkpt_dir: str) -> None:
        """
            Initializes the DuelingDeepNetwork class.

            Parameters:
                - learning_rate (float): The learning rate for the optimizer.
                - n_actions (int): The number of actions in the environment.
                - input_dims (list): The dimensions of the input state.
                - name (str): The name of the network.
                - chkpt_dir (str): The directory to save the network's checkpoints.

            Returns:
                - None
        """

        super(DuelingDeepNetwork, self).__init__()

        self.name = name

        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, self.name)
        
        self.fc = nn.Linear(*input_dims, 512)

        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        """
            Performs a forward pass on the network.

            Parameters:
                - state (T.Tensor): The input state.

            Returns:
                tuple[T.Tensor, T.Tensor]: The value and advantage outputs of the network.
        """

        x = self.fc(state)

        return self.value(x), self.advantage(x)

    def save_checkpoint(self) -> None:
        """
            Saves the network's checkpoint.

            Returns:
                - None
        """
        T.save(self.state_dict(), self.chkpt_file)

    def save_best(self, final_state: tuple) -> None:
        """
            Saves the network's checkpoint with the best score for a given final state.

            Parameters:
                - final_state (tuple): The final state of the environment.

            Returns:
                - None
        """

        T.save(
            self.state_dict(), 
            os.path.join(self.chkpt_dir, f'{self.name}_{final_state}')
        )

    def load_checkpoint(self) -> None:
        """
            Loads the network's checkpoint file.

            Returns:
                - None
        """

        print('Loading checkpoint...')
        self.load_state_dict(T.load(f'{self.chkpt_file}_best'))