import argparse
import os

import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.eager as tfe
# import tensorflow.contrib.eager as tfe
from preprocessing import preprocessing


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='~/tensorflowproject/data/mnistv',
                    help='The path to the CIFAR-10/CIFAR-100/mnist/mnist-fashion data directory.')

parser.add_argument('--train_dir', type=str, default='./log',
                    help='The directory where the model will be stored.')

parser.add_argument('--best_dir', type=str, default='./logbest',
                    help='The directory where the model will be stored.')

parser.add_argument('--dataset_name', type=str, default='mnist',
                    help='The dataset name.')

parser.add_argument('--model_name', type=str, default='fullnet',
                    help='The model name you choose.')

parser.add_argument('--train_epochs', type=int, default=100,
                    help='The number of epochs to train.')

parser.add_argument('--batch_size', type=int, default=100,
                    help='The number of images per batch.')

parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='The leatning rate.')

parser.add_argument('--normalization', type=str, default='NO',
                    help='The Normalization you choose, default Layer Normalization.')

parser.add_argument('--factor', type=float, default=0.1,
                    help='A multiplicative factor by which the values will be scaled.')

parser.add_argument('--l2_scale', type=float, default=0.0001,
                    help='The scale of L2 regularizer, 0.0 disables the L2 regularizer.')

parser.add_argument('--keep_probablity', type=float, default=0.9,
                    help='The keep probablity of dropout.')

parser.add_argument('--branches', type=int, default=2,
                    help='The branch number of dendritic neural network model.')

parser.add_argument('--layer_number', type=int, default=3,
                    help='The layer number of  network model.')

parser.add_argument('--net_length', type=int, default=512,
                    help='The unit number of  network model in every hidden layer.')

parser.add_argument('--branch_dir', type=str, default='./',
                    help='The directory where the information will be stored.')

FLAGS = parser.parse_args()


TRAIN_FILE = 'train.tfrecord'
VALIDATION_FILE = 'validation.tfrecord'
TEST_FILE = 'test.tfrecord'



def read_data(path, shape, file, datasetname, batch_size=100, is_train=False):
    filename = os.path.join(path, file)
    # print('-------------------data_dir:' + filename)
    # filename_queue = tf.data.Dataset.from_tensor_slices(filename)
    reader = tf.data.TFRecordDataset([filename])
    x = reader.shuffle(10).batch(10)
    print(x)
    j = reader.make_one_shot_iterator()
    i = j.get_next()
    features = tf.io.parse_single_example(i,
                               features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                        })
    if datasetname == 'mnist' or datasetname == 'fashion':
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        # img.set_shape(shape)
        img = tf.reshape(img, shape)
        img = tf.to_float(img)
        preprocessing_fn = preprocessing.get_preprocessing(datasetname, is_training=is_train)
        img = preprocessing_fn(img, 28, 28)
    elif datasetname == 'cifar10' or datasetname == 'cifar100':
        img = tf.decode_raw(features['img_raw'], tf.float32)
        img = tf.reshape(img, shape)
        image = tf.to_float(img)
        img = tf.image.per_image_standardization(img)
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    
    # with tf.Session() as sess:
    #     print("!!!!!!!!!!!!!!!!")
    #     label = tf.print(label, [label], message="This is x_test_: ")
    #     print("!!!!!!!!!!!!!!!!")
    return img, label



def main(unused_argv):
    num_classes = 10
    train_num_sample = 50000
    validation_num_sample = 5000

    if FLAGS.dataset_name == 'mnist':
        print("lol")
        train_file = 'train.tfrecord'
        validation_file = 'validation.tfrecord'
        # shape = [28 * 28]
        shape = [28, 28, 1]
        num_classes = 10
        train_num_sample = 55000
        validation_num_sample = 5000
        test_num_sample = 10000
        model_file_name = "model_mnist_tfrecord.ckpt"
        optimizer = tf.train.GradientDescentOptimizer
    
    else:
        raise ValueError("Give proper dataset name.")
    
    #create directory for training
    with tf.device('/cpu'):
        tl.files.exists_or_mkdir(FLAGS.train_dir)
    
        # read examples
        x_train_, y_train_ = read_data(path=FLAGS.data_dir,shape=shape,file=TRAIN_FILE, datasetname=FLAGS.dataset_name, is_train=True)
        x_validate_, y_validate_ = read_data(path=FLAGS.data_dir, shape=shape, file=VALIDATION_FILE, datasetname=FLAGS.dataset_name, is_train=False)
        x_test_, y_test_ = read_data(path=FLAGS.data_dir, shape=shape, file=TEST_FILE, datasetname=FLAGS.dataset_name, is_train=False)
        

        #shuffle examples
        batch_size = FLAGS.batch_size
        resume = False  # load model, resume from previous checkpoint?
        x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
            batch_size=batch_size, capacity=2000, min_after_dequeue=1000, num_threads=4) # set the number of threads here
        # for validation, uses batch instead of shuffle_batch
        x_validate_batch, y_validate_batch = tf.train.batch([x_validate_, y_validate_],
                                                    batch_size=batch_size, capacity=2000, num_threads=4)
        # for testing, uses batch instead of shuffle_batch
        x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
            batch_size=batch_size, capacity=2000, num_threads=4)
        
        tf.data.Dataset.shuffle(min_after_dequeue).batch(batch_size)

        print(x_train_)
        
        

        






if __name__ == '__main__':
    tfe.enable_eager_execution()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
