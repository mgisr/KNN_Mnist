from utils.mnist_format import *
from models.knn import KNN
from matplotlib import pyplot as plt
from PIL import Image


def main():
    (train_data, train_label), (test_data, test_label) = load_data()
    train_data = train_data.reshape((train_data.shape[0], -1))
    test_data = test_data.reshape((test_data.shape[0], -1))
    test_label = test_label.reshape((1, -1))
    tool = KNN()
    tool.fit(train_data, train_label)
    img = Image.fromarray(test_data[1].reshape((28, 28)))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
