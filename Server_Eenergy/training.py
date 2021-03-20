import os
import numpy as np
import random as rn

import environment
import brain
import dqn

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(85)
rn.seed(12345)

epsilon = 0.3 #our system will take 30% exploration(action random selection) and 70% explotation
number_actions = 5
direction_boundary = (number_actions -1)/2 #intermediate point (our boundary)
number_epochs = 100
max_memory = 3000
batch_size = 512
temp_step = 1.5


env = environment.Environment(optimal_temp = (15.0, 25.0), initial_month = 0, initial_number_users = 18, initial_rate_data = 32)

brain = brain.Brain(learning_rate = 0.00001, number_actions = number_actions)

dqn = dqn.DQN(max_memory = max_memory, discount_factor = 0.9)

train = True

env.train = train
model = brain.model
early_stopping = True
patience = 10
best_total_reward = -np.inf
patience_count = 0

if (env.train):
    # STARTING EPOCH BUCLE (1 Epoch = 5 Mouths)
    for epoch in range(1, number_epochs):
        # STARTING ENVIRONMEN VARIABLES AND TRAINING BUCLE
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0, 12)
        env.reset_env(new_month = new_month)
        game_over = False
        current_state, _, _ = env.give_env() #we only want current_state return from give_env method
        timestep = 0
        # INICIALIZATION TIMESTEPS BUCLE(Timestep = 1 minute) AT ONE EPOCH
        while ((not game_over) and timestep <= 5 * 30 * 24 * 60):
            # RUNNING NEXT ACTION BY EXPLORATION
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, number_actions)
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temp_step
            
            # RUNNING NEXT ACTION BY EXPLOTATION
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0])
                if (action - direction_boundary < 0):
                    direction = -1
                else:
                    direction = 1
                energy_ai = abs(action - direction_boundary) * temp_step
               
            # UPDATING ENVIRONMENT AND REACHING NEXT STATE
            next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep/(30*24*60)))
            total_reward += reward

            # SAVING NEW TRANSITION IN MEMORY
            dqn.remember([current_state, action, reward, next_state], game_over)

            # GETTING INPUTS AND TARGETS BLOCKS
            inputs, targets = dqn.get_batch(model, batch_size)

            # CALCULATING LOOST FUNCTION WITH THE WHOLE INPUT AND TARGET BLOCK
            loss += model.train_on_batch(inputs, targets)
            timestep += 1
            current_state = next_state

        # PRINTING RESULTS AT THE END OF EPOCH
        print("\n")
        print("Epoch: {:03d}/{:03d}.".format(epoch, number_epochs))
        print(" - Total Energy spended by IA: {:.0f} J.".format(env.total_energy_ai))
        print(" - Total Energy spended by no-IA: {:.0f} J.".format(env.total_energy_noai))

        # EARLY STOPPING
        if early_stopping:
            if (total_reward <= best_total_reward):
                patience_count += 1
            else:
                best_total_reward = total_reward
                patience_count = 0

            if patience_count >= patience:
                print("Early method execution.")
                break


        # Saving model for the future
        model.save("model.h5")

