import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI
import upypip as pip
#utility imports
from random import randint
import collections
import numpy as np

#part of structure taken from https://gist.github.com/arushir/04c58283d4fc00a4d6983dc92a3f1021
from dqn import DQN

class ExampleAPI(TFPluginAPI):

	def onSetup(self):
		pass

	def setupModel(self, jsonInput):
		#self.sess = tf.InteractiveSession()
		#self.graph = tf.get_default_graph()

		#self.x = tf.placeholder(tf.float32)
		
		#self.paddleY = tf.placeholder(tf.float32)
		#self.ballXY = tf.placeholder(tf.float32)
		#self.score = tf.placeholder(tf.float32)
		jsonArr = jsonInput.split(",")
		ue.log(str(jsonArr))
		self.iterations = 0

		DEFAULT_EPSILON = float(jsonArr[4])
		DECAY_RATE = float(jsonArr[5])
		EPSILON_MIN = float(jsonArr[6])

		DEFAULT_EPISODES = 2000
		DEFAULT_STEPS = 500 
		DEFAULT_ENVIRONMENT = 'BOT-UE4'

		DEFAULT_MEMORY_CAPACITY = 10000
		DEFAULT_GAMMA = 0.9
		DEFAULT_MINI_BATCH_SIZE = int(jsonArr[8])


		DEFAULT_LEARNING_RATE = float(jsonArr[7])
		DEFAULT_REGULARIZATION = 0.001
		DEFAULT_NUM_HIDDEN = 2 # not used in tensorflow implementation
		DEFAULT_HIDDEN_SIZE = int(jsonArr[9])
		DEFAULT_HIDDEN_SIZE2 = int(jsonArr[10])

		self.train_model = int(jsonArr[1])
		self.num_actions = int(jsonArr[3])

		self.agent_params = {'episodes': DEFAULT_EPISODES, 'steps': DEFAULT_STEPS, 'environment': DEFAULT_ENVIRONMENT, 'run_id': 1}
		self.cnn_params = {'lr': DEFAULT_LEARNING_RATE, 'reg': DEFAULT_REGULARIZATION, 'num_hidden':DEFAULT_NUM_HIDDEN, 'hidden_size':DEFAULT_HIDDEN_SIZE, 'hidden_size2':DEFAULT_HIDDEN_SIZE2,'mini_batch_size': DEFAULT_MINI_BATCH_SIZE}
		self.dqn_params = {'memory_capacity':DEFAULT_MEMORY_CAPACITY, 'epsilon':DEFAULT_EPSILON, 'gamma':DEFAULT_GAMMA,'mini_batch_size':DEFAULT_MINI_BATCH_SIZE, 'decay_rate': DECAY_RATE, 'epsilon_min': EPSILON_MIN}

		#use collections to manage a x frames buffer of input
		self.memory_capacity = 200
		self.inputQ = collections.deque(maxlen=self.memory_capacity)
		self.actionQ = collections.deque(maxlen=self.memory_capacity)

		null_input = np.zeros(int(jsonArr[2]))
		self.observation_shape = null_input.shape
		folder = jsonArr[0]
		self.model = DQN(self.num_actions, self.observation_shape, self.dqn_params, self.cnn_params, folder)

		#fill our deque so our input size is always the same
		for x in range(0, self.memory_capacity):
			self.inputQ.append(null_input)
			self.actionQ.append(0)

		return {'model created':True}
		
	#expected optional api: parse input object and return a result object, which will be converted to json for UE4
	def onJsonInput(self, jsonInput):

		action = randint(0, self.num_actions-1)
		#layer our input using deque ~200 frames so we can train with temporal data 

		#make a 1D stack of current input
		observation = jsonInput['percept']
		reward = jsonInput['reward']
		#ue.log("Percept: " + str(observation) + " reward: " + str(reward))

		#convert to list and set as x placeholder
		#feed_dict = {self.x: stackedList}
		#new_observation, reward, done, _ = env.step(action)

		#print(len(self.actionQ))
		lastAction = self.actionQ[self.memory_capacity-1]
		lastObservation = self.inputQ[self.memory_capacity-1]
		done = False
		
		# update the state 
		self.model.update_state(lastAction, lastObservation, observation, reward, done)

		# train step
		if(self.train_model == 1):
			self.model.train_step()
			#ue.log(str(observation))#Debug

		#append our stacked input to our deque
		self.inputQ.append(observation)
		#stackedList = list(self.inputQ)

		action = self.model.select_action(observation, self.iterations)
		self.actionQ.append(action)
		
        #counting iterations to save when we hit our memory
		self.iterations += 1

        #Calls saveBatchReward when we are at a completly new batch for plotting
		if(self.iterations%1000 == 0):
			self.saveBatchReward()

		#return selected action
		return {'action':float(action)}

	def saveModel(self, jsonInput):
	    self.model.model.saveModel(self.inputQ, self.actionQ)
	    pass
	def saveBatchReward(self):
	    self.model.saveBatchReward(self.iterations)
	    pass

	#expected optional api: start training your network
	def onBeginTraining(self):
		pass
	
#NOTE: this is a module function, not a class function. Change your CLASSNAME to reflect your class
#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return ExampleAPI.getInstance()