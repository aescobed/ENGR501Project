import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import math


# Environment example for testing
def env(parameters):
    target_values = [256, 128, 64]
    if parameters.ndim > 0:
        return 51 + sum(abs(target - param) for target, param in zip(target_values, parameters))
    else:
        return 51 + abs(target_values[0] - parameters)  # Fallback if not iterable


def generate_random_state():
    # Generates a random state with three parameters
    state = np.random.randint(0, 512, size=(1, 3))  # 1x3 matrix with random integers from 0 to 511
    return torch.from_numpy(state).float()  # Convert to a PyTorch tensor


# Neural network for Q-function approximation
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Epsilon-Greedy Strategy
def epsilon_greedy(state, epsilon, q_network):
    if random.random() < epsilon:
        # Generate a random action with the correct shape
        return torch.tensor([np.random.randint(0, 512, size=(3,))], dtype=torch.float).view(1, 3)
    else:
        with torch.no_grad():
            # Assuming the network outputs a tensor that can be interpreted as action values
            # Here, you need to make sure the network's output is used to select an action correctly
            action_values = q_network(state)
            # Example: If action_values is expected to be a vector of logits/actions
            # Select the max index but reshape correctly to match [1, 3]
            # This placeholder needs actual logic suitable for your setup:
            action = action_values.argmax(dim=1, keepdim=True)
            # Ensure it returns an appropriate action shape
            if action.shape[1] != 3:
                # Adjust action selection logic to create a tensor of shape [1, 3]
                # For now, let's assume we replicate the action index three times to match expected size
                action = action.repeat(1, 3)  # This is likely incorrect but serves as a placeholder
            return action

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Main training loop
def train_dqn():
    q_network = DQN()
    target_network = DQN()
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    # Learning rate - too high = divergence, too low = slow or local min
    LEARNING_RATE = 0.02

    # Weight decay - too high = underfitting, too low = overfitting (The depends on the nature of the poblem and whether the optimal parameter changes quickly)
    WEIGHT_DECAY = 0.1

    # Memory size - too high = could be learning from old experiences, too low = might not capture the diversity of the problem
    MEMORY_SIZE = 400

    # How long the agent will train
    NUM_EPISODES = 500

    # Epsilon start and start - start and end value for epsilon which decides how much the agent should be exploring
    EPS_START = 0.2
    EPS_END = 0.015

    # Epsilon decay - determines how quickly the agent transitions from exploring the environment randomly to exploiting what it has learned
    EPS_DECAY = 200

    # Batch size - Size of batches taken from experience replay
    BATCH_SIZE = 10

    # Determines the value of future rewards
    GAMMA = 0.6

    # Target update - Determines how frequently the weights for the NN are updated
    TARGET_UPDATE = 100

    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    memory = ReplayMemory(MEMORY_SIZE)
    steps_done = 0

    for episode in range(NUM_EPISODES):
        state = generate_random_state()
        for _ in range(500):  # Number of steps in each episode
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            action = epsilon_greedy(state, epsilon, q_network)
            reward = -env(action.squeeze().numpy()) # Ensure reward is a tensor

            next_state = generate_random_state()
            memory.push((state, action, next_state, torch.tensor([reward], dtype=torch.float).view(1, 1)))

            state = next_state

            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = tuple(zip(*transitions))

                # Debugging shapes:
                #for tensors in batch:
                #    print([t.shape for t in tensors])

                states, actions, next_states, rewards = [
                    torch.cat(tensors, dim=0) for tensors in batch
                ]
                state_action_values = q_network(states)
                next_state_values = target_network(next_states).max(1)[0].detach()
                expected_state_action_values = (next_state_values * GAMMA) + rewards

                loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            steps_done += 1

        print('episode: ', episode)
        print('States: ', states[0], ", ", states[1], ", ", states[2])
        print('Rewards: ', rewards[0], ", ", rewards[1], ", ", rewards[2])
        if episode % TARGET_UPDATE == 0:
            target_network.load_state_dict(q_network.state_dict())

    print('Training complete')


train_dqn()
