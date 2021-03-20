# IMPORT LIBRARIES 
import os
import numpy as np
import random as rn
from keras.models import load_model
import environment

# STTING UP SEEDS OF REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(85)
rn.seed(12345)

# SETTING UP PARAMETERS
number_actions = 5
direction_boundary = (number_actions -1)/2
temp_step = 1.5

# BUILDING ENVIERONMENT WITH A CLASS OBJECT ENMIRONMENT()
env = environment.Environment(optimal_temp = (15.0, 25.0), initial_month = 0, initial_number_users = 18, initial_rate_data = 32)

# UPLOADING THE MODEL
model = load_model("model.h5")

# CHOOSING THE TRAINING MODEL
train = False

# RUNNING A YEAR SIMULATION (EXPLOTATION)
env.train = train
current_state, _, _ = env.give_env()
for timestep in range(0, 12*30*24*60):
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])
            
    if (action < direction_boundary):
        direction = -1
    else:
        direction = 1
    energy_ai = abs(action - direction_boundary) * temp_step
    next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
    current_state = next_state


            
# PRINTING RESULTS AT THE END OF EPOCH
print("\n")
print("  - Total Energy spended by IA: {:.0f} J.".format(env.total_energy_ai))
print(" - Total Energy spended by no-IA: {:.0f} J.".format(env.total_energy_noai))
print("--------------------------------------------------")
print(" ENERGY SAVED: {:.0f} %.".format(100*(env.total_energy_noai-env.total_energy_ai)/env.total_energy_noai))
