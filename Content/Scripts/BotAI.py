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
		jsonArr = jsonInput.split(",")
		ue.log(str(jsonArr))
		self.iterations = 0

		DEFAULT_EPISODES = 2000
		DEFAULT_STEPS = 500 
		DEFAULT_ENVIRONMENT = 'BOT-UE4'

		DEFAULT_GAMMA = 0.9
		DEFAULT_REGULARIZATION = 0.001

		EPSILON = float(jsonArr[4])
		DECAY_RATE = float(jsonArr[5])
		EPSILON_MIN = float(jsonArr[6])

		MEMORY_CAPACITY = int(jsonArr[8])
		MINI_BATCH_SIZE = int(jsonArr[9])

		USE_DDQN = int(jsonArr[10])
		PRINT_OBS = int(jsonArr[11])
		PRINT_REWARD = int(jsonArr[12])
		USE_IMAGES = int(jsonArr[13])

		LEARNING_RATE = float(jsonArr[7])
		layer_amount = int(jsonArr[14])
		hidden_layers = []
		start = len(jsonArr) - layer_amount

		for i in range(layer_amount):
				hidden_layers.append(int(jsonArr[start+i]))

		self.train_model = int(jsonArr[1])
		self.num_actions = int(jsonArr[3])

		self.agent_params = {'episodes': DEFAULT_EPISODES, 'steps': DEFAULT_STEPS, 'environment': DEFAULT_ENVIRONMENT, 'run_id': 1}
		self.cnn_params = {'lr': LEARNING_RATE, 'reg': DEFAULT_REGULARIZATION,'hidden_layers':hidden_layers,'mini_batch_size': MINI_BATCH_SIZE,'use_images': USE_IMAGES}
		self.dqn_params = {'memory_capacity': MEMORY_CAPACITY,'epsilon': EPSILON,'gamma': DEFAULT_GAMMA,'mini_batch_size': MINI_BATCH_SIZE,'decay_rate': DECAY_RATE,'epsilon_min': EPSILON_MIN,'use_ddqn': USE_DDQN,'print_obs': PRINT_OBS,'print_reward': PRINT_REWARD,'use_images': USE_IMAGES}
		ue.log(str(self.dqn_params))
		ue.log(str(self.cnn_params))
		#use collections to manage a x frames buffer of input
		self.memory_capacity = 200
		self.inputQ = collections.deque(maxlen=self.memory_capacity)
		self.actionQ = collections.deque(maxlen=self.memory_capacity)

		null_input = np.zeros(int(jsonArr[2]))
		self.observation_shape = null_input.shape
		folder = jsonArr[0]
		self.model = DQN(self.num_actions, self.observation_shape, self.dqn_params, self.cnn_params, folder)
		self.mbs = DEFAULT_MINI_BATCH_SIZE

		#self.means = [45.10171524867664, 2515.6152058823905, 1.2756699421721667, 340.05399503555975, 1523.8039178279241, 1531.5614334074655, 1501.987002626419, 1480.2789580955505, 1466.3915354493458, 1485.4157070058186, 1488.6043882811864, 1478.684959236145, 1477.0602635631562, 1482.3246735661824, 1487.8680586878459, 1516.2443011096318, 1523.8039237890243, 0.8658333333333333, 8723.348632493336, 117.45349481709798, 9890.684453145344, 174.47253437042235, 9996.169818725586, 179.83003067334494]
		#self.sd = [2089.1286442845894, 2010277.4934772076, 10493.032393773728, 32043.8041775927, 2028527.2568337375, 2021956.1270421555, 1985549.9148059473, 1948318.4015361755, 1919737.4084812927, 1942997.8872622992, 1977754.6686598395, 1975682.4669874315, 1978499.8754247418, 1981972.5892711666, 1982403.9129913272, 2023941.1473031372, 2028527.2556589583, 4.018936833274738, 6919862.836880409, 15822.591556899479, 680448.762580585, 1699.2267270349653, 18223.64626244226, 48.01285669463955]
		self.means = [45.0717110799732, 2447.8929706197464, 3.74070544325926, 1490.4008984301431, 1456.354687444142, 1494.1618114403316, 1492.8436654570444, 1498.8283476292747, 1443.680688087191, 1490.4008905789512, 8735.875969903469, 117.2634552481515, 9887.840763203212, 174.2949338280814, 9997.659838378906, 179.89558674621583]
		self.sd = [2344.856102330476, 2011830.5842001913, 9800.033581368009, 2054003.769056988, 1967379.3959480862, 1990132.9755433355, 1958401.2946250923, 2013537.212580724, 1974069.5399232474, 2054003.7684572826, 6669498.576617268, 15913.078338011028, 698212.7759129503, 1763.2920645488573, 9251.814230709724, 29.339087941504033]

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
        #make observations into z-score
		for i in range(len(observation)):
				observation[i] = (observation[i]-self.means[i])/self.sd[i]

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

        #Cant start training before we have enough data in memory
		if(self.iterations == self.mbs+1):
			self.model.startTraining = True  

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