from smartsim import Experiment
from smartredis import ConfigOptions, Client
from smartredis import *
from smartredis.error import *
import numpy as np

REDIS_PORT = 6380

exp = Experiment("GPU_Optimizer", launcher="auto")

# Specify your C program executable instead of "program"
mpirun = exp.create_run_settings(
    "./program", run_command="mpirun"
)

multi_shard_config = ConfigOptions.create_from_environment("OPTIMIZER")

multi_shard_client = Client(multi_shard_config, logger_name="Model: multi shard logger")

# db = exp.create_database(db_nodes=1, port=REDIS_PORT, interface="lo")
# exp.generate(db)
# client = Client(address=db.get_address()[0], cluster=True)


#send_tensor = np.ones((4,3,3))
#client.put_tensor("tutorial_tensor_1", send_tensor)



mpirun.set_tasks(4)

model = exp.create_model("Simulation", mpirun)
exp.generate(model, overwrite=True)
exp.start(model, block=True, summary=True)

print(f"Model status: {exp.get_status(model)}")


retrieved_tensor = multi_shard_client.get_tensor("my_tensor")
print(retrieved_tensor)


