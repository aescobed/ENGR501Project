from smartsim import Experiment
from smartredis import ConfigOptions, Client
from smartredis import *
from smartredis.error import *
import numpy as np
import ML.model as model

REDIS_PORT = 6380

# Start experiment
exp = Experiment("GPU_Optimizer", launcher="auto")

# Start client
multi_shard_config = ConfigOptions.create_from_environment("OPTIMIZER")
multi_shard_client = Client(multi_shard_config, logger_name="Model: multi shard logger")





ML_params = model.SimpleParameters()
ML_params.reset()

print(ML_params.state[0] , "   " , ML_params.state[1])

# Set the parameters which the c program will use
param_in = np.array([[ML_params.state[0], ML_params.state[1], 1, 1],[0,0,0,0]])

# Send the tensor with the parameters to the smartredis database
multi_shard_client.put_tensor("parameters", param_in)


# Specify your C program executable
mpirun = exp.create_run_settings(
    "./program", run_command="mpirun"
)



mpirun.set_tasks(4)

model = exp.create_model("Simulation", mpirun)
exp.generate(model, overwrite=True)
exp.start(model, block=True, summary=True)

print(f"Model status: {exp.get_status(model)}")


retrieved_tensor = multi_shard_client.get_tensor("parameters")
print(retrieved_tensor)


