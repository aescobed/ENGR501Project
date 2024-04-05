import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class SimpleParameters:
    def __init__(self):
        self.action_space = 2  # Two parameters to optimize
        
        # First parameter is block size - maximum threads per block is 1024
        # Second parameter is the number of operations per GPU thread
        self.bounds = [[1, 1024] , [1, 100]]

    def reset(self):
        self.state = [random.uniform(*self.bounds[0]), random.uniform(*self.bounds[1])]
        return self.state


class DQN(nn.Module):

    # initialize the neural net
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    # forward propagation
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



def train_dqn(epochs=1):
    params = SimpleParameters()
    model = DQN(input_size=2, output_size=2)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    memory = deque(maxlen=2000)

    for epoch in range(epochs):
        state = torch.FloatTensor(env.reset()).unsqueeze(0)
        for t in range(200):
            if random.random() <= epsilon:
                action = [random.randrange(3), random.randrange(3)]
            else:
                with torch.no_grad():
                    action = model(state).argmax().item()
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32)

            memory.append((state, action, reward, next_state))

            if len(memory) > 128:
                batch = random.sample(memory, 128)
                batch_state, batch_action, batch_reward, batch_next_state = zip(*batch)

                batch_state = torch.cat(batch_state)
                batch_next_state = torch.cat(batch_next_state)
                batch_reward = torch.cat(batch_reward)

                current_q = model(batch_state)
                max_next_q = model(batch_next_state).max(1)[0]
                expected_q = batch_reward + gamma * max_next_q

                loss = loss_fn(current_q, expected_q.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Epsilon: {epsilon}")