import matplotlib
from utils.mnist_format import *
import matplotlib.pyplot as plt


def main():
    (train_data, train_label), (test_data, test_label) = load_data()
    plt.imshow(train_data[0])
    plt.show()
    print(train_label[0])


if __name__ == '__main__':
    main()
