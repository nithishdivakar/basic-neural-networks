import tensorflow as tf
tf.set_random_seed(1634)

# model
from model import Model

#data
from data import Data

F = Model()

D = Data("../data/mnist/test-images", batch_size = 10000)
# get all test data
x_train,y_train = D.get_batch(shuffle=False)
x_train = tf.reshape(x_train,[-1,28*28])

# forward computation.
y_logits, _ = F.inference(x_train)
correct_prediction = tf.equal(y_train, tf.cast(tf.argmax(y_logits, 1),dtype=tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess: 
  init = tf.global_variables_initializer()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  sess.run(init)
  # load latest saved model paramaters
  F.load_params(sess)
  print("Current model test ccuracy", sess.run(accuracy))
  coord.request_stop()
  coord.join(threads)
 
