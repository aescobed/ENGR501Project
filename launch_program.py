from smartsim import Experiment
from smartredis import Client
import numpy as np

REDIS_PORT = 6379

exp = Experiment("GPU_Optimizer", launcher="auto")

# Specify your C program executable instead of "program"
mpirun = exp.create_run_settings(
    "./program", run_command="mpirun"
)

db = exp.create_database(db_nodes=1, port=REDIS_PORT, interface="lo")
exp.generate(db)
exp.start(db)

client = Client(address=db.get_address()[0], cluster=False)

print("Address =", db.get_address()[0])

#send_tensor = np.ones((4,3,3))
#client.put_tensor("tutorial_tensor_1", send_tensor)



mpirun.set_tasks(4)

model = exp.create_model("Simulation", mpirun)
exp.generate(model, overwrite=True)
exp.start(model, block=True, summary=True)

print(f"Model status: {exp.get_status(model)}")


retrieved_tensor = client.get_tensor("my_tensor")
print(retrieved_tensor)

# Note: At this point, the C program ("my_c_program") will be executed as part of the experiment
# It should be designed to retrieve "tutorial_tensor_1", modify it, and perhaps save it back under a new key



# After the C program has run, you can retrieve and process the modified tensor as needed
# For example, if the C program saved the modified tensor with a new key:
