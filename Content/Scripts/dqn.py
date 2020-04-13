import numpy as np
import random as random
from collections import deque
import unreal_engine as ue
from PER import *
from cnn_tensorflow import CNN

# See https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf for model description

class DQN:
  def __init__(self, num_actions, observation_shape, dqn_params, cnn_params, folder):
    self.num_actions = num_actions
    self.observation_shape= observation_shape
    self.cnn_params = cnn_params
    self.folder = folder
    self.epsilon = dqn_params['epsilon']
    self.gamma = dqn_params['gamma']
    self.mini_batch_size = dqn_params['mini_batch_size']
    self.time_step = 0
    self.decay_rate = dqn_params['decay_rate']
    self.epsilon_min = dqn_params['epsilon_min']
    self.current_epsilon = self.epsilon
    self.startTraining = False

    # memory 
    self.memory = deque(maxlen=dqn_params['memory_capacity'])

    #PER memory
    self.per_memory = Memory(dqn_params['memory_capacity'])

    # initialize network
    self.model = CNN(folder, num_actions, observation_shape, cnn_params)
    print("model initialized")
    self.target_model = CNN(folder, num_actions, observation_shape, cnn_params)

  def select_action(self, observation, iterations):
    """
    Selects the next action to take based on the current state and learned Q.
    Args:
      observation: the current state
    """
    if(iterations%1000==0):
        self.current_epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.decay_rate * self.time_step)
        self.time_step += 1
        #ue.log(str(self.current_epsilon))
        
    if random.random() < self.current_epsilon:
      # with epsilon probability select a random action 
      action = np.random.randint(0, self.num_actions)
    else:
      # select the action a which maximizes the Q value
      obs = np.array([observation])
      q_values = self.model.predict(obs)
      action = np.argmax(q_values)

    return action

  def update_state(self, action, observation, new_observation, reward, done):
    """
    Stores the most recent action in the replay memory.
    Args: 
      action: the action taken 
      observation: the state before the action was taken
      new_observation: the state after the action is taken
      reward: the reward from the action
      done: a boolean for when the episode has terminated 
    """
    transition = {'action': action,
                  'observation': observation,
                  'new_observation': new_observation,
                  'reward': reward,
                  'is_done': done}
    self.memory.append(transition)
    self.per_memory.store(transition)

  def get_random_mini_batch(self):
    """
    Gets a random sample of transitions from the replay memory.
    """
    rand_idxs = random.sample(range(len(self.memory)), self.mini_batch_size)
    mini_batch = []
    for idx in rand_idxs:
      mini_batch.append(self.memory[idx])

    return mini_batch

  def train_step(self):
    """
    Updates the model based on the mini batch
    """
    if self.startTraining == True:
      #mini_batch = self.get_random_mini_batch()
      tree_idx, mini_batch  = self.per_memory.sample(self.mini_batch_size)
      #ue.log(str(self.observation_shape[0]))
      new_states = np.zeros((self.mini_batch_size, self.observation_shape[0]))
      old_states = np.zeros((self.mini_batch_size, self.observation_shape[0]))
      actionsBatch = []

      for i in range(self.mini_batch_size):
        new_states[i] = mini_batch[i]['new_observation']
        old_states[i] = mini_batch[i]['observation']
        actionsBatch.append(mini_batch[i]['action'])
        
      target = self.model.predict(old_states)
      target_old = np.array(target)
      target_next = self.model.predict(new_states)
      target_val = self.target_model.predict(new_states)
      #ue.log(str(target_next_test))
      Xs = []
      ys = []
      actions = []

      for i in range(self.mini_batch_size):
        y_j = mini_batch[i]['reward']
        # for nonterminals, add gamma*max_a(Q(phi_{j+1})) term to y_j
        if not mini_batch[i]['is_done']:
          #old_observation = mini_batch[0]['observation']
          #old_obs = np.array([old_observation])
          #target = self.model.predict(old_obs)

          #new_observation = sample['new_observation']
          #new_obs = np.array([new_observation])
          #q_new_values = self.model.predict(new_obs)#target next
          q_new_values = target_next[i]
          #target_val = self.target_model.predict(new_obs)

          action = np.max(q_new_values)
          actionIndex = np.argmax(q_new_values)

          #y_j += self.gamma*action
          y_j += self.gamma*target_val[i][actionIndex]

        action = np.zeros(self.num_actions)
        action[mini_batch[i]['action']] = 1

        observation = mini_batch[i]['observation']

        Xs.append(observation.copy())
        ys.append(y_j)
        actions.append(action.copy())

      #Seting up for training
      Xs = np.array(Xs)
      ys = np.array(ys)
      actions = np.array(actions)

      #Updateing PER bintree
      indices = np.arange(self.mini_batch_size, dtype=np.int32)
      #ue.log(str(ys))
      actionsInt = np.array(actionsBatch, dtype=int)
      #ue.log(str(actionsInt))
      absolute_errors = np.abs(target_old[indices, actionsInt]-ys[indices])
      #ue.log(str(absolute_errors))
      # Update priority
      self.per_memory.batch_update(tree_idx, absolute_errors)


      #absolute_errors = np.abs(target_old - target)outside sample?
      #self.per_memory.batch_update(tree_idx, absolute_errors)

      

      self.model.train_step(Xs, ys, actions)


      self.model.train_step(Xs, ys, actions)

  def saveBatchReward(self, iterations):
    
    #Need this on set iteration update
    self.target_model = CNN(self.folder, self.num_actions, self.observation_shape, self.cnn_params)

    r = 0
    it = iterations/1000
    index = 0
    for x in range((len(self.memory))-1000, (len(self.memory))):
      t = self.memory[x]
      r += t['reward']
    file = self.model.model_directory + "/plot.txt"
    try:
      f = open(file, "r")
      lines = f.read().splitlines()
      last_line = lines[-1]
      #ue.log(str(last_line))
      index = int(str(last_line.split(",")[0])) + 1
      #ue.log(index)
      f.close()
    except:
      index = 1
    ue.log("Saved: " + str(index) +  "," + str(r) + " memlen: " + str(len(self.memory)) + ", Epislon: " + str(self.current_epsilon))
    f = open(file, "a+")
    f.write(str(index)+ "," + str(r) + "\n")
    f.close()
    pass