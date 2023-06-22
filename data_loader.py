import numpy as np
import struct
from array import array
from os.path  import join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
Define a class for loading the MNIST dataset from the files
Source: https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
'''
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        '''
        Set the files path
        TRAIN_IMAGES_FILE_PATH: Path to the train images
        training_labels_filepath: Path to the train labels
        test_images_filepath: Path to the test images
        test_labels_filepath: Path to the test labels
        '''
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):
        '''
        Read the images and labels
        images_filepath: Path to the images
        labels_filepath: Path to the labels
        Return: The numpy array containing the images and labels
        '''        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28 *28)
            images[i][:] = img            
        
        return np.array(images), np.array(labels)
            
    def load_data(self):
        '''
        Load the train and test images, train and test labels from the file paths
        '''
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)
    

# Define the paths of the files
INPUT_PATH = 'data'
TRAIN_IMAGES_FILE_PATH = join(INPUT_PATH, 'train-images.idx3-ubyte')
TRAIN_LABELS_FILE_PATH = join(INPUT_PATH, 'train-labels.idx1-ubyte')
TEST_IMAGES_FILE_PATH = join(INPUT_PATH, 'test-images.idx3-ubyte')
TEST_LABELS_FILE_PATH = join(INPUT_PATH, 'test-labels.idx1-ubyte')

# Create MNISTDataloader and load the images
mnist = MnistDataloader(
    training_images_filepath=TRAIN_IMAGES_FILE_PATH,
    training_labels_filepath=TRAIN_LABELS_FILE_PATH,
    test_images_filepath=TEST_IMAGES_FILE_PATH,
    test_labels_filepath=TEST_LABELS_FILE_PATH
)

# Split into train and test set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Split into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, stratify=y_train)

# Standardize the train, validation and test dataset
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)