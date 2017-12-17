# addapted from github.com/kwotsin/create_tfrecords/create_tfrecords.py

from dataset_utils import *
import tensorflow as tf
import os
import argparse



def make_dataset(data_source = os.path.join(os.getcwd(), 'data'), data_out_dir=os.path.join(os.getcwd(), 'data'),
                 num_shards = 10, tfrecord_filename = 'webgazer'):

    if dataset_exists(dataset_dir=data_out_dir, num_shards=num_shards, output_filename=tfrecord_filename):
        print('Dataset files already exist.')
        return None

    print('Dataset files do not yet exist - making files\n')

    if not os.path.exists(data_out_dir):
        os.mkdir(data_out_dir)
        print('Dataset directory did not exist made new directory: %s' % data_out_dir)

    # to filter out the empty directories in the dataset
    def dir_is_empty(path):
        path = os.path.join(data_source, path)
        return len(os.listdir(path)) > 0

    with tf.gfile.Open(os.path.join(data_source, 'train_1430_1.txt'), 'r') as f:
        list_of_dirs_train = f.read()
    list_of_dirs_train = list_of_dirs_train.split('\n')
    list_of_dirs_train = list(filter(None, list_of_dirs_train))
    list_of_dirs_train = [os.path.normpath(dir_name) for dir_name in list_of_dirs_train]
    list_of_dirs_train = [dir_name.replace('\\', '/') for dir_name in list_of_dirs_train]
    list_of_dirs_train = list(filter(dir_is_empty, list_of_dirs_train))

    train_image_filenames = get_filenames(data_source, list_of_dirs_train)
    train_image_labels = make_label_dict(data_source, list_of_dirs_train)
    train_image_eye_centers = get_eye_centers(data_source, list_of_dirs_train)

    print('Done reading train\n')

    with tf.gfile.Open(os.path.join(data_source, 'test_1430_1.txt'), 'r') as f:
        list_of_dirs_test = f.read()
    list_of_dirs_test = list_of_dirs_test.split('\n')
    list_of_dirs_test = list(filter(None, list_of_dirs_test))
    list_of_dirs_test = [os.path.normpath(dir_name) for dir_name in list_of_dirs_test]
    list_of_dirs_test = [dir_name.replace('\\', '/') for dir_name in list_of_dirs_test]
    list_of_dirs_test = list(filter(dir_is_empty, list_of_dirs_test))

    test_image_filenames = get_filenames(data_source, list_of_dirs_test)
    test_image_labels = make_label_dict(data_source, list_of_dirs_test)
    test_image_eye_centers = get_eye_centers(data_source, list_of_dirs_train)

    print('Done reading test\n')

    convert_dataset(split_name='train', filenames=train_image_filenames, labels=train_image_labels,
                    eye_centers=train_image_eye_centers, dataset_dir=data_out_dir,
                    tfrecord_filename=tfrecord_filename, num_shards=num_shards)

    print('Done converting training data to TFRecord files\n')

    convert_dataset(split_name='test', filenames=test_image_filenames, labels=test_image_labels,
                    eye_centers=test_image_eye_centers, dataset_dir=data_out_dir,
                    tfrecord_filename=tfrecord_filename, num_shards=num_shards)

    print('Done converting test data to TFRecord files\n')


    print('\nFinished converting the %s dataset!' % (tfrecord_filename) )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Accept command line arguments to name dataset, '\
                                                 + 'choose a location to save it, and determine '\
                                                 + 'how many files is is spread across')
    parser.add_argument('-data_source', '--data_source', default=os.path.join(os.getcwd(), 'data'), type=str,
                        help='path where data files are read from')
    parser.add_argument('-data_out_dir', '--data_out_dir', default=os.path.join(os.getcwd(), 'data'), type=str,
                        help='path where dataset files will be saved')
    parser.add_argument('-num_shards', '--num_shards', default=10, type=int,
                        help='number of files to split each of test and train data across  - int')
    parser.add_argument('-data_filename', '--data_filename', default='webgazer', type=str,
                        help='string with the base name of dataset files')

    args = parser.parse_args()
    args = vars(args)

    make_dataset(data_source=args['data_source'], data_out_dir=args['data_out_dir'],
                 num_shards=args['num_shards'], tfrecord_filename=args['data_filename'])
