from smartsim import Experiment
from smartredis import ConfigOptions, Client
from smartredis import *
from smartredis.error import *
import numpy as np
import ML.model as ML
from ML.model import MLModel
import torch.optim as optim
import torch
import math
import torch.nn as nn
import logging

logging.basicConfig(filename='training_data.log', level=logging.DEBUG, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

REDIS_PORT = 6380

# Start experiment
exp = Experiment("GPU_Optimizer", launcher="auto")

# Start client
multi_shard_config = ConfigOptions.create_from_environment("OPTIMIZER")
multi_shard_client = Client(multi_shard_config, logger_name="Model: multi shard logger")


model = MLModel()
PM = ML.ModelPMs()
optimizer = optim.Adam(model.q_network.parameters(), lr=PM.LEARNING_RATE, weight_decay=PM.WEIGHT_DECAY)
memory = ML.ReplayMemory(PM.MEMORY_SIZE)

#Start with large reward for comparison
old_time = 999999
for episode in range(PM.NUM_EPISODES):
    
    steps_done = 0
    state = ML.generate_random_state()
    for iteration in range(200):  # Number of steps in each episode
        epsilon = PM.EPS_END + (PM.EPS_START - PM.EPS_END) * math.exp(-1. * steps_done / PM.EPS_DECAY)
        action = ML.epsilon_greedy(state, epsilon, model.q_network)

        next_state = ML.apply_action_to_state(state, action)

        # Set the parameters which the c program will use
        param_in = np.array([[next_state[0, 0], next_state[0, 1], next_state[0, 2]],[0,0,0]])

        # Send the tensor with the parameters to the smartredis database
        multi_shard_client.put_tensor("parameters", param_in)

        # Specify your C program executable
        mpirun = exp.create_run_settings(
            "./program", run_command="mpirun"
        )
        mpirun.set_tasks(next_state[0,2])
        #mpirun.set_tasks(4)

        SimulationModel = exp.create_model("Simulation", mpirun)
        exp.generate(SimulationModel, overwrite=True)
        exp.start(SimulationModel, block=True, summary=True)
        #print(f"Model status: {exp.get_status(model)}")

        # Get the results from the experiment
        retrieved_tensor = multi_shard_client.get_tensor("parameters")
        new_time = retrieved_tensor[1,0]

        #print("new time: ", new_time)

        reward = ML.GetReward(old_time, new_time)
        if torch.all(state == next_state):
            reward = 0

        memory.push((state, action, next_state, torch.tensor([reward], dtype=torch.float).view(1, 1)))

        state = next_state
        old_time = new_time

        if len(memory) >= PM.BATCH_SIZE:
            transitions = memory.sample(PM.BATCH_SIZE)
            batch = tuple(zip(*transitions))


            #for tensors in batch:
            #    print([t.shape for t in tensors])

            states, actions, next_states, rewards = [
                torch.cat(tensors, dim=0) for tensors in batch
            ]
            model.q_network(states).shape
            current_q_values = model.q_network(states).gather(1, actions)
            max_next_q_values = model.target_network(next_states).max(1)[0].detach()
            max_next_q_values = max_next_q_values.clone().detach()
            expected_q_values = (PM.GAMMA * max_next_q_values)
            
            rewards = rewards.clone().detach()
            rewards = rewards.squeeze()
            expected_q_values = expected_q_values + rewards


            # Compute loss
            loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        steps_done += 1

        if episode % PM.TARGET_UPDATE == 0:
            model.target_network.load_state_dict(model.q_network.state_dict())

        if iteration % 10 ==0:
            logging.info("Episode: %d, Step: %d", episode, iteration)
        
        logging.info("Inputs: [%d, %d, %d], Program Time: %.8f", state[0,0], state[0,1], state[0,2], retrieved_tensor[1,0])




