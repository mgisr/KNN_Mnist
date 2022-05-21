import numpy as np
import os, gzip, pickle


def load_mnist(data_file, label_file):
    with gzip.open(data_file, 'rb') as f:
        magic_num = int(f.read(4).hex(), 16)  # 读取出的四个字节是16进制，需要转成10进制
        image_num = int(f.read(4).hex(), 16)
        image_width = int(f.read(4).hex(), 16)
        image_height = int(f.read(4).hex(), 16)
        img_data = np.frombuffer(f.read(), dtype='uint8')  # 将剩余所有数据一次读取至numpy数组中
        img_data = img_data.reshape((image_num, image_width, image_height))

    with gzip.open(label_file, 'rb') as f:
        magic_num = int(f.read(4).hex(), 16)
        label_num = int(f.read(4).hex(), 16)
        label_data = np.frombuffer(f.read(), dtype='uint8')

    return img_data, label_data


def load_data():
    if not os.path.exists('data/mnist.pkl'):
        train_data, train_labels = load_mnist('data/mnist/train-images-idx3-ubyte.gz', 'data/mnist/train-labels-idx1-ubyte.gz')
        test_data, test_labels = load_mnist('data/mnist/t10k-images-idx3-ubyte.gz', 'data/mnist/t10k-labels-idx1-ubyte.gz')
        dataset = {'train_data': train_data, 'train_labels': train_labels, 'test_data': test_data,
                   'test_labels': test_labels}
        with open('mnist.pkl', 'wb') as f:
            pickle.dump(dataset, f)
    with open('data/mnist.pkl', 'rb') as f:
        dataset = pickle.load(f)
    return (dataset['train_data'], dataset['train_labels']), (dataset['test_data'], dataset['test_labels'])
