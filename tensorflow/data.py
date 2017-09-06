import os
import glob
import tensorflow as tf

class Data(object):
  '''
    This class handles data. It expects the data to be in the following format
    root_dir
    |- class_one
    |   |- data
    |   |- data
    |   ...
    |-class_two
    |   |- data
    |   ...
    ....
    
  '''
  def __init__(self, root_path_name,batch_size, file_ext="png"):
    self.root_dir = root_path_name
    self.class_names = sorted([path for path in os.listdir(self.root_dir) 
                        if os.path.isdir(os.path.join(self.root_dir,path))])
    self.batch_size = batch_size

    self.datum_paths = []   # list of paths of datums
    self.datum_labels = []  # list containing labels
    for class_idx, class_name in enumerate(self.class_names):
      for file_name in glob.glob(os.path.join(self.root_dir,class_name)+"/*."+file_ext):
        self.datum_paths.append(file_name)
        self.datum_labels.append(class_idx)

    return
  
  def _get_one(self, input_queue_slice):
    '''
      Logic for reading one file.
    '''
    # print filename_queue
    datum_file_path = input_queue_slice[0]
    class_idx       = input_queue_slice[1]

    image_reader = tf.WholeFileReader()
    datum        = tf.read_file(datum_file_path)

    image = tf.image.decode_image(datum, channels=1)
    image = tf.reshape(image, [28,28,1])
    image = tf.to_float(image)
   
    label = class_idx

    return image, label

  def get_batch(self, num_epochs=None,shuffle=True): 
    input_queue = tf.train.slice_input_producer(
                     [self.datum_paths, self.datum_labels], 
                     num_epochs=num_epochs, 
                     shuffle=shuffle
                  )

    datum, label  = self._get_one(input_queue)
    
    datum_batch, label_batch = tf.train.batch([datum, label], batch_size=self.batch_size, capacity = 100)
    # (X,Y)
    return datum_batch, label_batch



if __name__ == '__main__':

  EI = Data("../data/mnist/test-images",batch_size = 2)
  
  x_,y_ = EI.get_minibatch_tensors()

  with tf.Session() as sess:
    tf.global_variables_initializer()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    x,y = sess.run([x_,y_])
    print( x[0,:,:,0],y[0])
    print( x[0].shape)
    coord.request_stop()
    coord.join(threads)

