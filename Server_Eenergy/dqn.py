import numpy as np

class DQN(object):

    # INICIALIZATION DQN PARAMETERS
    def __init__(self, max_memory = 100, discount_factor = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount_factor = discount_factor #gamma factor

    # BUILDING A EXPERIENCE DELAY MEMORY
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # BUILDING A METHOD THAT BUILDS TWO INPUTS AND TARGETS BLOCKS AND EXTRACTING THE MEMORY'S TRANSITIONS
    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1]
        inputs = np.zeros((min(batch_size, len_memory), num_inputs))
        targets = np.zeros((min(batch_size, len_memory), num_outputs))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[idx][0] #we want the memory's transition
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sa = np.max(model.predict(next_state)[0])
            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount_factor*Q_sa
        return inputs, targets
