import utils
import dataset

'''
线性核：['l', c, kernel_weight] —— c * x1 @ x2
高斯核：['g', c, sigma, kernel_weight] —— c * exp(- |x1 - x2| ** 2 / (2 * sigma ** 2))
多项式核：['p', c, degree, theta, kernel_weight] —— c * (x1 @ x2 + theta) ** degree
sigmoid核：['s', c, beta, theta, kernel_weight] —— c * tanh(beta * x1 @ x2 + theta)
'''

ml = ['l', 0.1, 1]
mg = ['g', 10, 5, 1]
mp = ['p', 1, 2, 2, 1]
ms = ['s', 1, 0.01, -1, 1]

cl = ['l', 0.5, 1]
cg = ['g', 10, 1, 1]
cp = ['p', 1, 4, 0.5, 1]
cs = ['s', 1, 0.05, -1, 1]

'''
kernel_type: list —— 选择核函数类型
'''
kernel_type = [ml, ms]

# 导入数据集
cifar10, mnist = dataset.load_dataset()
cifar10_train_data, cifar10_test_data, cifar10_test_labels = cifar10
mnist_train_data, mnist_test_data, mnist_test_labels = mnist

# 使用MNIST数据集
'''
test_train: bool —— True: 测试训练集，False: 不测试训练集
method: str —— 'OVO': OVO策略，'OVA': OVA策略
'''
mnist_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False, method='OVO')

# 使用CIFAR-10数据集
cifar10_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False, method='OVO')