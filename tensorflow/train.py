import tensorflow as tf
tf.set_random_seed(1634)

import sys

# model
from model import Model

#data
from data import Data

# losses
from loss import cross_entropy_loss

# params
lr = 1e-5
total_updates = 10000
log_interval  = 1000

F = Model()
D = Data("../data/mnist/train-images",batch_size = 32)


x_train,y_train = D.get_batch()
x_train = tf.reshape(x_train,[-1,28*28])

opt = tf.train.GradientDescentOptimizer(lr)

y_logits, _ = F.inference(x_train)
loss      = cross_entropy_loss(logits = y_logits, labels = y_train)

correct_prediction = tf.equal(y_train, tf.cast(tf.argmax(y_logits, 1),dtype=tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

back_prop = opt.minimize(loss)

with tf.Session() as sess: 
  # tensor flow things
  init = tf.global_variables_initializer()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  
  sess.run(init)
  for i in range(1,total_updates+1):
    sess.run(
      [back_prop],
      feed_dict = {}
    )
    if not i % log_interval:
      F.save_params(sess,i)
      print(i,sess.run(loss),sess.run(accuracy))        
      sys.stdout.flush()

    '''
    if step % args.snapshot_interval == 0:
      saver.save(sess, "model.ckpt")
    '''
  coord.request_stop()
  coord.join(threads)
  
