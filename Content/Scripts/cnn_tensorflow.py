import tensorflow as tf
import numpy as np
import logging
import unreal_engine as ue

class CNN:
  """
  Convolutional Neural Network model.
  """

  def __init__(self, num_actions, observation_shape, params={}, verbose=False):
    """
    Initialize the CNN model with a set of parameters.
    Args:
      params: a dictionary containing values of the models' parameters.
    """
    self.scripts_path = ue.get_content_dir() + "Scripts"
    self.model_directory = self.scripts_path + "/model"

    self.modemodel_loaded = False

    self.model_path = self.model_directory + "/model.ckpt"
    self.verbose = verbose
    self.num_actions = num_actions

    # observation shape will be a tuple
    self.observation_shape = observation_shape[0]
    logging.info('Initialized with params: {}'.format(params))

    self.lr = params['lr']
    self.reg = params['reg']
    self.num_hidden = params['num_hidden']
    self.hidden_size = params['hidden_size']

    self.session = self.create_model()


  def add_placeholders(self):
    input_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_shape))
    labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
    actions_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_actions))

    return input_placeholder, labels_placeholder, actions_placeholder


  def nn(self, input_obs):
    with tf.name_scope("Layer1") as scope:
      W1shape = [self.observation_shape, self.hidden_size]
      W1 = tf.get_variable("W1", shape=W1shape,)
      bshape = [1, self.hidden_size]
      b1 = tf.get_variable("b1", shape=bshape, initializer = tf.constant_initializer(0.0))

    with tf.name_scope("Layer2") as scope:
      W2shape = [self.hidden_size, self.hidden_size]
      W2 = tf.get_variable("W2", shape=W2shape,)
      bshape = [1, self.hidden_size]
      b2 = tf.get_variable("b2", shape=bshape, initializer = tf.constant_initializer(0.0))

    with tf.name_scope("OutputLayer") as scope:
      Ushape = [self.hidden_size, self.num_actions]
      U = tf.get_variable("U", shape=Ushape)
      b3shape = [1, self.num_actions]
      b3 = tf.get_variable("b3", shape=b3shape, initializer = tf.constant_initializer(0.0))

    xW = tf.matmul(input_obs, W1)
    h = tf.tanh(tf.add(xW, b1))

    xW = tf.matmul(h, W2)
    h = tf.tanh(tf.add(xW, b2))

    hU = tf.matmul(h, U)    
    out = tf.add(hU, b3)

    reg = self.reg * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)) + tf.reduce_sum(tf.square(U)))
    return out, reg

  def loadnn(self, input_placeholder):
    saver = tf.train.import_meta_graph(self.model_path + ".meta")
    ue.log('meta graph imported')
    saver.restore(session, tf.train.latest_checkpoint(self.model_directory))
    ue.log('graph restored')
    self.model_loaded = True

    #restore our weights
    self.graph = tf.get_default_graph()
    W1 = self.graph.get_tensor_by_name("W1:0")
    b1 = self.graph.get_tensor_by_name("b1:0")
    W2 = self.graph.get_tensor_by_name("W2:0")
    b2 = self.graph.get_tensor_by_name("b2:0")
    U = self.graph.get_tensor_by_name("U:0")
    b3 = self.graph.get_tensor_by_name("b3:0")

    xW = tf.matmul(self.input_placeholder, W1)
    h = tf.tanh(tf.add(xW, b1))

    xW = tf.matmul(h, W2)
    h = tf.tanh(tf.add(xW, b2))

    hU = tf.matmul(h, U)    
    outputs = tf.add(hU, b3)

    reg = self.reg * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)) + tf.reduce_sum(tf.square(U)))
    return outputs, reg
  def create_model(self):
    """
    The model definition.
    """
    session = tf.Session()
    ue.log('creating new model')
    self.input_placeholder, self.labels_placeholder, self.actions_placeholder = self.add_placeholders()
    outputs, reg = self.nn(self.input_placeholder)
    self.predictions = outputs
    
    self.q_vals = tf.reduce_sum(tf.multiply(self.predictions, self.actions_placeholder), 1)

    self.loss = tf.reduce_sum(tf.square(self.labels_placeholder - self.q_vals)) + reg

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)

    self.train_op = optimizer.minimize(self.loss)
    init = tf.initialize_all_variables()
    self.saverino = tf.train.Saver()
    session.run(init)
    ue.log('new session created')
    return session

  def train_step(self, Xs, ys, actions):
    """
    Updates the CNN model with a mini batch of training examples.
    """

    loss, _, prediction_probs, q_values = self.session.run(
      [self.loss, self.train_op, self.predictions, self.q_vals],
      feed_dict = {self.input_placeholder: Xs,
                  self.labels_placeholder: ys,
                  self.actions_placeholder: actions
                  })

  def predict(self, observation):
    """
    Predicts the rewards for an input observation state. 
    Args:
      observation: a numpy array of a single observation state
    """

    loss, prediction_probs = self.session.run(
      [self.loss, self.predictions],
      feed_dict = {self.input_placeholder: observation,
                  self.labels_placeholder: np.zeros(len(observation)),
                  self.actions_placeholder: np.zeros((len(observation), self.num_actions))
                  })

    return prediction_probs

  def saveModel(self, inputQ, actionQ):
    path = self.saverino.save(self.session, self.model_path)
    ue.log("Saved model yes: "+str(path))
    pass