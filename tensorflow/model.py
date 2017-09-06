import tensorflow as tf

class Model(object):
  def __init__(self):
    self.PARAMS = {}
    self._init_params()
    self.saver = tf.train.Saver(self.PARAMS)

  def _init_params(self):
    def add_param(shape,name):
      self.PARAMS[name] = tf.Variable(
                           tf.truncated_normal(
			     shape = shape, 
			     mean=0.0,stddev=0.01
			   ), 
			   name = name
			 )
    
    add_param([28*28,100],"W1")
    add_param([100],"b1")
    add_param([  100,10 ],"W2")
    add_param([ 10],"b2")
    

  def inference(self,in_tensor):
    # helper functions
    def dense(x,w,b):
      wx    = tf.matmul(x, w)
      wxpb  = tf.nn.bias_add(wx,b)
      return wxpb

    # forward computation
    z1 = dense(in_tensor,self.PARAMS["W1"],self.PARAMS["b1"])
    a1 = tf.nn.relu(z1)
    z2 = dense(a1,self.PARAMS["W2"],self.PARAMS["b2"])
    a2 = tf.nn.softmax(z2)

    return a2,z2


  def save_params(self,sess,step=None):
    self.saver.save(sess,"model", global_step = step)
  
  def load_params(self,sess):
    self.saver.restore(sess,tf.train.latest_checkpoint('./'))
