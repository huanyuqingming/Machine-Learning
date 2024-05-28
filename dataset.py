import numpy as np
import pickle
import tarfile
import gzip
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
from skimage import data, exposure, color

# 解压
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 加载数据集
def load_cifar10(cifar10_path):
    with tarfile.open(cifar10_path) as tar:
        tar.extractall()
        train_data = []
        train_labels = []
        for i in range(1, 6):
            batch_data = unpickle(f'cifar-10-batches-py/data_batch_{i}')
            train_data.extend(batch_data[b'data'].reshape(-1, 3, 32, 32))
            train_labels += batch_data[b'labels']
        test_batch = unpickle('cifar-10-batches-py/test_batch')
        test_data = test_batch[b'data'].reshape(-1, 3, 32, 32)
        test_labels = np.array(test_batch[b'labels'])
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)

    # 对标签0-9分别取前500个样本做训练集
    train_data = np.concatenate([train_data[train_labels == i][:500] for i in range(10)], axis=0)
    train_labels = np.concatenate([np.full(500, i) for i in range(10)], axis=0)
    # 对标签0-9分别取前100个样本做测试集
    test_data = np.concatenate([test_data[test_labels == i][:100] for i in range(10)], axis=0)
    test_labels = np.concatenate([np.full(100, i) for i in range(10)], axis=0)

    return train_data, train_labels, test_data, test_labels
    
def load_mnist(mnist_paths):
    for path in mnist_paths:
        with gzip.open(path, 'rb') as f:
            if 'images' in path:
                if 'train' in path:
                    train_data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
                else:
                    test_data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            else:
                if 'train' in path:
                    train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
                else:
                    test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # 对标签0-9分别取前500个样本做训练集
    train_data = np.concatenate([train_data[train_labels == i][:500] for i in range(10)], axis=0)
    train_labels = np.concatenate([np.full(500, i) for i in range(10)], axis=0)
    # 对标签0-9分别取前500个样本做测试集
    test_data = np.concatenate([test_data[test_labels == i][:100] for i in range(10)], axis=0)
    test_labels = np.concatenate([np.full(100, i) for i in range(10)], axis=0)
    
    return train_data, train_labels, test_data, test_labels
    
    
# 导入数据集
def load_dataset():
    cifar10_path = r"dataset\cifar-10-python.tar.gz"
    tmp_train_data, train_labels, test_data, test_labels = load_cifar10(cifar10_path)
    train_data = []
    for i in range(10):
        train_data.append(tmp_train_data[train_labels == i])
    cifar10 = [np.array(train_data) / 255, np.array(test_data) / 255, test_labels]

    mnist_paths = [r'dataset\t10k-images-idx3-ubyte.gz', r'dataset\t10k-labels-idx1-ubyte.gz', r'dataset\train-images-idx3-ubyte.gz', r'dataset\train-labels-idx1-ubyte.gz']
    tmp_train_data, train_labels, test_data, test_labels = load_mnist(mnist_paths)
    train_data = []
    for i in range(10):
        train_data.append(tmp_train_data[train_labels == i])
    mnist = [np.array(train_data) / 255, np.array(test_data) / 255, test_labels]

    return cifar10, mnist

# 展平数据集
def flatten_dataset(train_data, test_data):
    train_data_reshaped = []
    test_data_reshaped = []
    for i in range(10):
        train_data_reshaped.append(train_data[i].reshape(len(train_data[i]), -1))
    test_data_reshaped = test_data.reshape(len(test_data), -1)

    return train_data_reshaped, test_data_reshaped

# 提取颜色直方图
def extract_color_histogram(train_data, test_data):
    train_data_hist = []
    test_data_hist = []
    for i in range(10):
        train_data_hist.append(np.array([np.histogram(train_data[i][j].flatten(), bins=256, range=(0, 1))[0] for j in range(len(train_data[i]))]))
    test_data_hist = np.array([np.histogram(test_data[j].flatten(), bins=256, range=(0, 1))[0] for j in range(len(test_data))])

    return train_data_hist, test_data_hist


# 提取HOG特征
def extract_hog(train_data, test_data):
    train_data_hog = []
    test_data_hog = []
    for i in range(10):
        train_data_hog.append(np.array([hog(train_data[i][j], pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=0) for j in range(len(train_data[i]))]))
    test_data_hog = np.array([hog(test_data[j], pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=0) for j in range(len(test_data))])

    return train_data_hog, test_data_hog

def visualize_hog(image):
    image = np.transpose(image, (1, 2, 0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('Input image')

    
    image = color.rgb2gray(image)
    # 提取HOG特征
    # 设置visualize=True
    fd, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    # 使用 skimage.exposure.rescale_intensity 函数来增强图像的对比度
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # 显示原始图像和HOG特征图像


    

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    plt.show()


# 数据集可视化
def visualize(cifar10, mnist):
    cifar10_train_data, cifar10_test_data, cifar10_test_labels = cifar10
    mnist_train_data, mnist_test_data, mnist_test_labels = mnist

    print(f'CIFAR-10 train data: {len(cifar10_train_data)*len(cifar10_train_data[0])}')
    print(f'CIFAR-10 test data: {len(cifar10_test_data)}')
    print(f'MNIST train data: {len(mnist_train_data)*len(mnist_train_data[0])}')
    print(f'MNIST test data: {len(mnist_test_data)}')

    cifar10_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig, axs = plt.subplots(1, 4)
    a = random.randint(0, 9)
    b = random.randint(0, 499)
    c = random.randint(0, 999)
    axs[2].imshow(cifar10_train_data[a][b].transpose((1, 2, 0)))
    axs[2].set_title(f'CIFAR-10: {cifar10_class[a]}')
    axs[3].imshow(cifar10_test_data[c].transpose((1, 2, 0)))
    axs[3].set_title(f'CIFAR-10: {cifar10_class[cifar10_test_labels[c]]}')
    axs[0].imshow(mnist_train_data[a][b], cmap='gray')
    axs[0].set_title(f'MNIST: {a}')
    axs[1].imshow(mnist_test_data[c], cmap='gray')
    axs[1].set_title(f'MNIST: {mnist_test_labels[c]}')
    plt.show()


if __name__ == '__main__':
    cifar10, mnist = load_dataset()
    visualize_hog(cifar10[0][7][237])
    visualize(cifar10, mnist)