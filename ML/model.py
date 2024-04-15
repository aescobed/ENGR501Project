import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import math



# Environment example for testing
def env(prevState, nextState):
    target_values = [256, 128, 64]

    nextState = np.atleast_2d(nextState)
    #prevState = torch.tensor(prevState, dtype=torch.float32)
    #nextState = torch.tensor(nextState, dtype=torch.float32)
    rewards = []

    prevState = np.array(prevState, dtype=np.float32)
    nextState = np.array(nextState.squeeze(), dtype=np.float32)

    #for prevParam, nextParam, target in zip(prevState, nextState, target_values):
    
    for prevParam, nextParam, target in zip(prevState, nextState, target_values):
        diff =  np.abs(prevParam-target) - np.abs(nextParam-target)
        if diff > 0:
            rewards.append(1)
        elif diff < 0:
            rewards.append(-1)
        else:
            rewards.append(0)
        
    rewardIndx = np.argmax(np.abs(rewards))
    return rewards[rewardIndx]


def apply_action_to_state(state, action):
    action_effects = np.array([
        [2, 0, 0],  # Action 0: Increase first parameter
        [-2, 0, 0],  # Action 1: Decrease first parameter
        [0, 2, 0],  # Action 2: Increase second parameter
        [0, -2, 0],  # Action 3: Decrease second parameter
        [0, 0, 2],  # Action 4: Increase third parameter
        [0, 0, -2]   # Action 5: Decrease third parameter
    ])
    return torch.tensor(state + action_effects[action], dtype=torch.float)
     



def generate_random_state():
    state = np.random.randint(0, 512, size=(1, 3))  # 1x3 matrix with random integers from 0 to 511
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

# Main training loop
def train_dqn():
    q_network = DQN()

    # target network is not updated as often
    target_network = DQN()
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    # Learning rate - too high = divergence, too low = slow or local min
    LEARNING_RATE = 0.0001

    # Weight decay - too high = underfitting, too low = overfitting (The depends on the nature of the poblem and whether the optimal parameter changes quickly)
    WEIGHT_DECAY = 0.0001

    # Memory size - too high = could be learning from old experiences, too low = might not capture the diversity of the problem
    MEMORY_SIZE = 5000

    # How long the agent will train
    NUM_EPISODES = 5000

    # Epsilon start and start - start and end value for epsilon which decides how much the agent should be exploring
    EPS_START = 0.2
    EPS_END = 0.015

    # Epsilon decay - determines how quickly the agent transitions from exploring the environment randomly to exploiting what it has learned
    EPS_DECAY = 2

    # Batch size - Size of batches taken from experience replay
    BATCH_SIZE = 10

    # Determines the value of future rewards
    GAMMA = 0.9

    # Target update - Determines how frequently the weights for the NN are updated
    TARGET_UPDATE = 100

    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    memory = ReplayMemory(MEMORY_SIZE)
    steps_done = 0

    for episode in range(NUM_EPISODES):
        steps_done = 0
        state = generate_random_state()
        for _ in range(500):  # Number of steps in each episode
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            action = epsilon_greedy(state, epsilon, q_network)
            next_state = apply_action_to_state(state, action)
            

            #reward = -env(next_state.detach().numpy())
            reward = env(state.squeeze(), next_state.squeeze())


            
            memory.push((state, action, next_state, torch.tensor([reward], dtype=torch.float).view(1, 1)))

            state = next_state

            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = tuple(zip(*transitions))


                #for tensors in batch:
                #    print([t.shape for t in tensors])

                states, actions, next_states, rewards = [
                    torch.cat(tensors, dim=0) for tensors in batch
                ]
                q_network(states).shape
                current_q_values = q_network(states).gather(1, actions)
                max_next_q_values = target_network(next_states).max(1)[0].detach()
                max_next_q_values = max_next_q_values.clone().detach()
                expected_q_values = (GAMMA * max_next_q_values)
                
                rewards = rewards.clone().detach()
                rewards = rewards.squeeze()
                expected_q_values = expected_q_values + rewards


                # Compute loss
                loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            steps_done += 1


        if episode % 10 == 0:
            print('episode: ', episode)
            #print('Q values: ', current_q_values)
            print('Rewards: ', rewards)
            #print('Actions: ', actions)
            print('State: ', state)
            print('Loss: ', loss.item())


        if episode % TARGET_UPDATE == 0:
            target_network.load_state_dict(q_network.state_dict())
            

    print('Training complete')


train_dqn()
