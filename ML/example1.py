import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class SimpleFunctionEnv:
    def __init__(self):
        self.x = 0

    def step(self, action):
        step_size = 0.1
        if action == 0:
            self.x -= step_size
        else:
            self.x += step_size

        reward = -(self.x - 2) ** 2
        next_state = self.x
        done = True if abs(self.x - 2) < 0.1 else False
        return next_state, reward, done

    def reset(self):
        self.x = random.uniform(0, 4)
        return self.x

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

env = SimpleFunctionEnv()
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

episodes = 500
epsilon = 1.0  # Starting value of epsilon
epsilon_decay = 0.99  # Decay rate for epsilon
min_epsilon = 0.01  # Minimum value of epsilon

for episode in range(episodes):
    state = env.reset()
    done = False
    total_loss = 0
    while not done:
        state_tensor = torch.FloatTensor([state]).unsqueeze(0)
        q_values = model(state_tensor)
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)  # Explore
        else:
            action = q_values.argmax().item()  # Exploit

        next_state, reward, done = env.step(action)
        q_value = q_values[0, action].unsqueeze(0)
        next_state_tensor = torch.FloatTensor([next_state]).unsqueeze(0)
        next_q_values = model(next_state_tensor)
        max_next_q_value = torch.tensor([reward + 0.99 * next_q_values.max().item()], requires_grad=True)

        loss = loss_fn(q_value, max_next_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        total_loss += loss.item()

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    #if episode % 100 == 0:
    print(f'Episode {episode}, Loss: {total_loss / (episode + 1)}, Epsilon: {epsilon}')

# Test the trained model
test_state = env.reset()
state_tensor = torch.FloatTensor([test_state]).unsqueeze(0)
q_values = model(state_tensor)
action = q_values.argmax().item()
print(f'Test State: {test_state}, Action: {"Increase" if action else "Decrease"}')

