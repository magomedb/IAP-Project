import tensorflow as tf
import numpy as np
import logging
import unreal_engine as ue
import upypip as pip
from PIL import Image

class CNN:
  """
  Convolutional Neural Network model.
  """

  def __init__(self, folder, num_actions, observation_shape, params={}, verbose=False):
    """
    Initialize the CNN model with a set of parameters.
    Args:
      params: a dictionary containing values of the models' parameters.
    """
    self.scripts_path = ue.get_content_dir() + "Scripts"
    self.model_directory = self.scripts_path + "/models" + "/" + folder

    self.modemodel_loaded = False

    self.model_path = self.model_directory + "/model.ckpt"
    self.verbose = verbose
    self.num_actions = num_actions

    # observation shape will be a tuple
    self.observation_shape = observation_shape[0]
    logging.info('Initialized with params: {}'.format(params))

    self.use_images = params['use_images']
    self.lr = params['lr']
    self.reg = params['reg']
    self.hidden_layers = params['hidden_layers']
    self.conv_layers = params['conv_layers']
    self.image_width = params['image_width']
    self.image_height = params['image_height']
    self.color_channels = params['color_channels']
    self.W = []
    self.b = []
    self.conv = []
    self.conv_kernels = []
    self.conv_biases = []
    self.session = self.create_model()


  def add_placeholders(self):
    input_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_shape))
    labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
    actions_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_actions))

    return input_placeholder, labels_placeholder, actions_placeholder

  def nn(self, input_obs):
    ue.log(str('CNN created.'))
    input_obs = tf.reshape(input_obs, shape=[-1, self.image_height, self.image_width, self.color_channels])

    with tf.name_scope("ConvolutionalLayer1") as scope:
      #current_activation = 'tf.nn.' + self.conv_layers[0][4]
      conv1 = tf.layers.conv2d(inputs = input_obs, filters = int(self.conv_layers[0][0]), kernel_size = int(self.conv_layers[0][1]), strides = int(self.conv_layers[0][2]), padding = self.conv_layers[0][3], activation = tf.nn.relu, name="conv1")
      conv1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 1)
      self.conv.append(conv1)
      #ue.log('Values: ' + str(self.conv_layers[0][0]) + ', ' + str(self.conv_layers[0][1]) + ', ' + str(self.conv_layers[0][2]) + ', ' + str(self.conv_layers[0][3]) + ', ' + str(self.conv_layers[0][4]))
      #ue.log(str(len(self.conv_layers)))
      #ue.log(str(self.conv[0]))
      #self.conv.append(input_obs)

    for i in range(len(self.conv_layers)-1):
        scopeName = "ConvolutionalLayer" + str(i+2)
        convName = "conv" + str(i+2)
        with tf.name_scope(scopeName) as scope:
            ue.log(str(i))
            current_filters = int(self.conv_layers[i+1][0])
            current_kernels = int(self.conv_layers[i+1][1])
            current_strides = int(self.conv_layers[i+1][2])
            current_padding = self.conv_layers[i+1][3]
            #current_activation = 'tf.nn.' + self.conv_layers[i+1][4]   # Need to account for activation being None
            conv = tf.layers.conv2d(inputs = self.conv[i], filters = current_filters, kernel_size = current_kernels, strides = current_strides, padding = current_padding, activation = tf.nn.relu, name=convName)
            conv = tf.layers.max_pooling2d(inputs = conv, pool_size = [2,2], strides = 1)
            self.conv.append(conv)
            #ue.log(str(self.conv[i]))
            #ue.log('Values: ' + str(current_filters) + ', ' + str(current_kernels) + ', ' + str(current_strides) + ', ' + str(current_padding) + ', ' + str(current_activation))

    
    #scopeName = "ConvolutionalLayer" + str(len(self.conv_layers))
    #convName = "conv" + str(len(self.conv_layers))

    #with tf.name_scope(scopeName) as scope:
    #    current_filters = int(self.conv_layers[len(self.conv_layers)-1][0])
    #    current_kernels = int(self.conv_layers[len(self.conv_layers)-1][1])
    #    current_strides = int(self.conv_layers[len(self.conv_layers)-1][2])
    #    current_padding = self.conv_layers[len(self.conv_layers)-1][3]
        #current_activation
    #    conv = tf.layers.conv2d(inputs = self.conv[i], filters = current_filters, kernel_size = current_kernels, strides = current_strides, padding = current_padding, activation = tf.nn.relu, name=convName)
        #conv = tf.layers.max_pooling2d(inputs = conv, pool_size = [2,2], strides = 2)
    #    self.conv.append(conv)

    finalConv = tf.contrib.layers.flatten(self.conv[len(self.conv)-1])

    with tf.name_scope("Layer1") as scope:
      W1shape = [finalConv.shape[1], self.hidden_layers[0]]
      self.W.append(tf.get_variable("W1", shape=W1shape,))
      b1shape = [1, self.hidden_layers[0]]
      self.b.append(tf.get_variable("b1", shape=b1shape, initializer = tf.constant_initializer(0.0)))

    for i in range(len(self.hidden_layers)-1):
        scopeName = "Layer" + str(i+2)
        WName = "W" + str(i+2)
        bName = "b" + str(i+2)
        with tf.name_scope(scopeName) as scope:
            Wshape = [self.hidden_layers[i], self.hidden_layers[i+1]]
            self.W.append(tf.get_variable(WName, shape=Wshape,))
            bshape = [1, self.hidden_layers[i+1]]
            self.b.append(tf.get_variable(bName, shape=bshape, initializer = tf.constant_initializer(0.0)))
    
    scopeName = "Layer" + str(len(self.hidden_layers)+1)
    WName = "W" + str(len(self.hidden_layers)+1)
    bName = "b" + str(len(self.hidden_layers)+1)

    with tf.name_scope(scopeName) as scope:
      Wshape = [self.hidden_layers[len(self.hidden_layers)-1], self.hidden_layers[len(self.hidden_layers)-1]]
      self.W.append(tf.get_variable(WName, shape=Wshape,))
      b4shape = [1, self.hidden_layers[len(self.hidden_layers)-1]]
      self.b.append(tf.get_variable(bName, shape=b4shape, initializer = tf.constant_initializer(0.0)))

    with tf.name_scope("OutputLayer") as scope:
      Ushape = [self.hidden_layers[len(self.hidden_layers)-1], self.num_actions]
      self.U = tf.get_variable("U", shape=Ushape)
      boshape = [1, self.num_actions]
      self.bo = tf.get_variable("bo", shape=boshape, initializer = tf.constant_initializer(0.0))

    xW = tf.matmul(finalConv, self.W[0])
    h = tf.tanh(tf.add(xW, self.b[0]))
    regCalc = tf.reduce_sum(tf.square(self.W[0]))
    for i in range(len(self.W)-1):
        xW = tf.matmul(h, self.W[i+1])
        h = tf.tanh(tf.add(xW, self.b[i+1]))
        regCalc += tf.reduce_sum(tf.square(self.W[i+1]))

    hU = tf.matmul(h, self.U)
    out = tf.add(hU, self.bo)
    regCalc += tf.reduce_sum(tf.square(self.U))

    reg = self.reg * regCalc
    ue.log('model values created')
    #self.conv_kernels.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv1/kernel')[0])

    return out, reg

  def dnn(self, input_obs):
    with tf.name_scope("Layer1") as scope:
      W1shape = [self.observation_shape, self.hidden_layers[0]]
      self.W.append(tf.get_variable("W1", shape=W1shape,))
      b1shape = [1, self.hidden_layers[0]]
      self.b.append(tf.get_variable("b1", shape=b1shape, initializer = tf.constant_initializer(0.0)))

    for i in range(len(self.hidden_layers)-1):
        scopeName = "Layer" + str(i+2)
        WName = "W" + str(i+2)
        bName = "b" + str(i+2)
        with tf.name_scope(scopeName) as scope:
            Wshape = [self.hidden_layers[i], self.hidden_layers[i+1]]
            self.W.append(tf.get_variable(WName, shape=Wshape,))
            bshape = [1, self.hidden_layers[i+1]]
            self.b.append(tf.get_variable(bName, shape=bshape, initializer = tf.constant_initializer(0.0)))
    
    scopeName = "Layer" + str(len(self.hidden_layers)+1)
    WName = "W" + str(len(self.hidden_layers)+1)
    bName = "b" + str(len(self.hidden_layers)+1)
    with tf.name_scope(scopeName) as scope:
      Wshape = [self.hidden_layers[len(self.hidden_layers)-1], self.hidden_layers[len(self.hidden_layers)-1]]
      self.W.append(tf.get_variable(WName, shape=Wshape,))
      b4shape = [1, self.hidden_layers[len(self.hidden_layers)-1]]
      self.b.append(tf.get_variable(bName, shape=b4shape, initializer = tf.constant_initializer(0.0)))

    with tf.name_scope("OutputLayer") as scope:
      Ushape = [self.hidden_layers[len(self.hidden_layers)-1], self.num_actions]
      self.U = tf.get_variable("U", shape=Ushape)
      boshape = [1, self.num_actions]
      self.bo = tf.get_variable("bo", shape=boshape, initializer = tf.constant_initializer(0.0))

    xW = tf.matmul(input_obs, self.W[0])
    h = tf.tanh(tf.add(xW, self.b[0]))
    regCalc = tf.reduce_sum(tf.square(self.W[0]))
    for i in range(len(self.W)-1):
        xW = tf.matmul(h, self.W[i+1])
        h = tf.tanh(tf.add(xW, self.b[i+1]))
        regCalc += tf.reduce_sum(tf.square(self.W[i+1]))

    hU = tf.matmul(h, self.U)
    out = tf.add(hU, self.bo)
    regCalc += tf.reduce_sum(tf.square(self.U))

    reg = self.reg * regCalc
    #ue.log(str(W1))
    #ue.log(str(b1))
    #ue.log(str(W2))
    #ue.log(str(b2))
    #ue.log(str(W3))
    #ue.log(str(b3))
    #ue.log(str(U))
    #ue.log(str(b4))
    ue.log('model values created')
    return out, reg

#not used
  def loadnn(self, input_placeholder, sess):
    model_loaded = False
    out = None
    reg = None
    with sess.as_default():
      try:
        saver = tf.train.import_meta_graph(self.model_path + ".meta")
        ue.log('meta graph imported')
        saver.restore(sess, tf.train.latest_checkpoint(self.model_directory))
        ue.log('graph restored')
        model_loaded = True
        #restore our weights
        self.graph = tf.get_default_graph()
        self.W1 = self.graph.get_tensor_by_name("W1:0")
        self.b1 = self.graph.get_tensor_by_name("b1:0")
        self.W2 = self.graph.get_tensor_by_name("W2:0")
        self.b2 = self.graph.get_tensor_by_name("b2:0")
        self.W3 = self.graph.get_tensor_by_name("W3:0")
        self.b3 = self.graph.get_tensor_by_name("b3:0")
        self.U = self.graph.get_tensor_by_name("U:0")
        self.b4 = self.graph.get_tensor_by_name("b4:0")

        xW = tf.matmul(input_placeholder, self.W1)
        h = tf.tanh(tf.add(xW, self.b1))

        xW = tf.matmul(h, self.W2)
        h = tf.tanh(tf.add(xW, self.b2))

        xW = tf.matmul(h, self.W3)
        h = tf.tanh(tf.add(xW, self.b3))

        hU = tf.matmul(h, self.U)
        out = tf.add(hU, self.b4)

        reg = self.reg * (tf.reduce_sum(tf.square(self.W1)) + tf.reduce_sum(tf.square(self.W2)) + tf.reduce_sum(tf.square(self.W3)) + tf.reduce_sum(tf.square(self.U)))
        ue.log('Session variables restored')
        #ue.log(str(W1))
        #ue.log(str(b1))
        #ue.log(str(W2))
        #ue.log(str(b2))
        #ue.log(str(W3))
        #ue.log(str(b3))
        #ue.log(str(U))
        #ue.log(str(b4))
      except:
          model_loaded = False
    return out, reg, model_loaded

  def create_model(self):
    """
    The model definition.
    """
    tf.reset_default_graph()
    session = tf.Session()

    self.input_placeholder, self.labels_placeholder, self.actions_placeholder = self.add_placeholders()
    outputs, reg = self.nn(self.input_placeholder) if self.use_images == 1 else self.dnn(self.input_placeholder)

    self.predictions = outputs

    self.q_vals = tf.reduce_sum(tf.multiply(self.predictions, self.actions_placeholder), 1)

    self.loss = tf.reduce_sum(tf.square(self.labels_placeholder - self.q_vals)) + reg

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.lr)

    self.train_op = optimizer.minimize(self.loss)

    self.saverino = tf.train.Saver()
    try:
        saver = tf.train.Saver()
        saver.restore(session, self.model_path)
        ue.log("model restored")
        #ue.log(str(session.run(self.W2)))#test values
    except:
        init = tf.initialize_all_variables()
        self.saverino = tf.train.Saver()
        session.run(init)
        ue.log('Created new model')

    ue.log('session created')
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
    #ue.log(str(self.session.run(self.W2)))#test values
    ue.log("Saved model: "+str(self.model_path))
    pass

