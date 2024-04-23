import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import math




def apply_action_to_state(state, action):
    action_effects = np.array([
        [10, 0, 0],  # Action 0: Increase first parameter
        [-10, 0, 0],  # Action 1: Decrease first parameter
        [0, 1, 0],  # Action 2: Increase second parameter
        [0, -1, 0],  # Action 3: Decrease second parameter
        [0, 0, 1],  # Action 4: Increase third parameter
        [0, 0, -1]   # Action 5: Decrease third parameter
    ])

    # Apply action effect to create the new state candidate
    new_state = state + action_effects[action]

    # Ensure new state values do not exceed boundaries
    new_state[0, 0] = min(max(new_state[0, 0], 1), 1024)
    new_state[0, 1] = min(max(new_state[0, 1], 0), 50)
    new_state[0, 2] = min(max(new_state[0, 2], 1), 4)


    return torch.tensor(new_state, dtype=torch.float)
     



def generate_random_state():
    state = np.zeros((1,3))
    state[0, 0] = np.random.randint(1, 1025)  # parameter for block size
    state[0, 1] = np.random.randint(0, 51)  # parameter for number of repeats
    state[0, 2] = np.random.randint(1, 5)  # parameter for number of mpi threads
    return torch.from_numpy(state).float()  # Convert to a PyTorch tensor


# Neural network for Q-function approximation
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def epsilon_greedy(state, epsilon, q_network):
    if random.random() > epsilon:
        with torch.no_grad():
            return q_network(state).max(1)[1].view(1, 1)  # Exploitation: Choose the best action based on max Q-value
    else:
        return torch.tensor([[random.randrange(6)]], dtype=torch.long)  


# The experience of the NN
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ModelPMs:
    # Learning rate - too high = divergence, too low = slow or local min
    LEARNING_RATE = 0.0001

    # Weight decay - too high = underfitting, too low = overfitting (The depends on the nature of the poblem and whether the optimal parameter changes quickly)
    WEIGHT_DECAY = 0.0001

    # Memory size - too high = could be learning from old experiences, too low = might not capture the diversity of the problem
    MEMORY_SIZE = 500

    # How long the agent will train
    NUM_EPISODES = 500

    # Epsilon start and start - start and end value for epsilon which decides how much the agent should be exploring
    EPS_START = 0.3
    EPS_END = 0.015

    # Epsilon decay - determines how quickly the agent transitions from exploring the environment randomly to exploiting what it has learned
    EPS_DECAY = 10

    # Batch size - Size of batches taken from experience replay
    BATCH_SIZE = 50

    # Determines the value of future rewards
    GAMMA = 0.9

    # Target update - Determines how frequently the weights for the NN are updated
    TARGET_UPDATE = 100





class MLModel:
    def __init__(self):
        self.q_network = DQN()

        # target network is not updated as often
        self.target_network = DQN()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.state = generate_random_state()



def GetReward(old_out, new_out):
    if old_out > new_out:
        return 1

    else:
        return -1




if __name__ == '__main__':
    train_dqn()
