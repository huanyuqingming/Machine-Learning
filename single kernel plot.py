import utils
import dataset
import matplotlib.pyplot as plt


'''
线性核：['l', c, kernel_weight] —— c * x1 @ x2
高斯核：['g', c, sigma, kernel_weight] —— c * exp(- |x1 - x2| ** 2 / (2 * sigma ** 2))
多项式核：['p', c, degree, theta, kernel_weight] —— c * (x1 @ x2 + theta) ** degree
sigmoid核：['s', c, beta, theta, kernel_weight] —— c * tanh(beta * x1 @ x2 + theta)
'''
kernel_type = ['s']
use_dataset = 'cifar10'

# 导入数据集
cifar10, mnist = dataset.load_dataset()
cifar10_train_data, cifar10_test_data, cifar10_test_labels = cifar10
mnist_train_data, mnist_test_data, mnist_test_labels = mnist

# 参数消融
# 使用MNIST数据集
# 线性核
if use_dataset == 'mnist':
    if kernel_type[0] == 'l':
        mnist_test_accs = {}
        for c in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10]:
            print(f"MNIST dataset, Linear kernel, C={c}")
            kernel_type = [['l', c, 1]]
            mnist_test_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False)
            mnist_test_accs[c] = mnist_test_acc
        # 绘制参数消融结果
        plt.figure()
        plt.plot(mnist_test_accs.keys(), mnist_test_accs.values(), '-o', label="Test Accuracy")
        plt.xlabel("C")
        plt.ylabel("Accuracy")
        plt.title("MNIST dataset, Linear kernel")
        plt.legend()
        plt.gca().set_xscale('log') 
        plt.show()

    # 高斯核
    elif kernel_type[0] == 'g':
        mnist_test_accs = {}
        for c in [0.1, 0.5, 1, 5, 10]:
            for sigma in [0.5, 0.7, 1, 3, 5, 7, 10, 13, 15, 17, 20]:
                print(f"MNIST dataset, Gaussian kernel, C={c}, Sigma={sigma}")
                kernel_type = [['g', c, sigma, 1]]
                mnist_test_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False)
                mnist_test_accs[(c, sigma)] = mnist_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (c, sigma), acc in mnist_test_accs.items():
            if c not in data:
                data[c] = {'sigma': [], 'acc': []}
            data[c]['sigma'].append(sigma)
            data[c]['acc'].append(acc)
        # 对每个c值绘制一条曲线
        plt.figure()
        for c, values in data.items():
            plt.plot(values['sigma'], values['acc'], '-o', label=f"C={c}")
        plt.xlabel("Sigma")
        plt.ylabel("Accuracy")
        plt.title("MNIST dataset, Gaussian kernel")
        plt.legend()
        plt.gca().set_xscale('log') 
        plt.show()

    # 多项式核
    elif kernel_type[0] == 'p':
        mnist_test_accs = {}
        for degree in [1, 2, 3, 4, 5]:
            for theta in [0, 0.5, 1, 2, 3, 4, 5]:
                print(f"MNIST dataset, Polynomial kernel, Degree={degree}, Theta={theta}")
                kernel_type = [['p', 1, degree, theta, 1]]
                mnist_test_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False)
                mnist_test_accs[(degree, theta)] = mnist_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (degree, theta), acc in mnist_test_accs.items():
            if degree not in data:
                data[degree] = {'theta': [], 'acc': []}
            data[degree]['theta'].append(theta)
            data[degree]['acc'].append(acc)
        # 对每个degree值绘制一条曲线
        plt.figure()
        for degree, values in data.items():
            plt.plot(values['theta'], values['acc'], '-o', label=f"Degree={degree}")
        plt.xlabel("Theta")
        plt.ylabel("Accuracy")
        plt.title("MNIST dataset, Polynomial kernel")
        plt.legend()
        plt.show()

    # sigmoid核
    elif kernel_type[0] == 's':
        mnist_test_accs = {}
        for beta in [1e-3, 5e-3, 1e-2, 5e-2]:
            for theta in [-3, -2, -1, 0, 1, 2, 3]:
                print(f"MNIST dataset, Sigmoid kernel, Beta={beta}, Theta={theta}")
                kernel_type = [['s', 1, beta, theta, 1]]
                mnist_test_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False)
                mnist_test_accs[(beta, theta)] = mnist_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (beta, theta), acc in mnist_test_accs.items():
            if beta not in data:
                data[beta] = {'theta': [], 'acc': []}
            data[beta]['theta'].append(theta)
            data[beta]['acc'].append(acc)
        # 对每个beta值绘制一条曲线
        plt.figure()
        for beta, values in data.items():
            plt.plot(values['theta'], values['acc'], '-o', label=f"Beta={beta}")
        plt.xlabel("Theta")
        plt.ylabel("Accuracy")
        plt.title("MNIST dataset, Sigmoid kernel")
        plt.legend()
        plt.show()

    else:
        print("Invalid kernel type!")
        exit(1)

# 使用CIFAR-10数据集
elif use_dataset == 'cifar10':
    if kernel_type[0] == 'l':
        cifar10_test_accs = {}
        for c in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10]:
            print(f"CIFAR-10 dataset, Linear kernel, C={c}")
            kernel_type = [['l', c, 1]]
            cifar10_test_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False)
            cifar10_test_accs[c] = cifar10_test_acc
        # 绘制参数消融结果
        plt.figure()
        plt.plot(cifar10_test_accs.keys(), cifar10_test_accs.values(), '-o', label="Test Accuracy")
        plt.xlabel("C")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 dataset, Linear kernel")
        plt.legend()
        plt.gca().set_xscale('log') 
        plt.show()

    # 高斯核
    elif kernel_type[0] == 'g':
        cifar10_test_accs = {}
        for c in [0.1, 0.5, 1, 5, 10]:
            for sigma in [0.5, 0.7, 1, 3, 5, 7, 10, 13, 15, 17, 20]:
                print(f"CIFAR-10 dataset, Gaussian kernel, C={c}, Sigma={sigma}")
                kernel_type = [['g', c, sigma, 1]]
                cifar10_test_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False)
                cifar10_test_accs[(c, sigma)] = cifar10_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (c, sigma), acc in cifar10_test_accs.items():
            if c not in data:
                data[c] = {'sigma': [], 'acc': []}
            data[c]['sigma'].append(sigma)
            data[c]['acc'].append(acc)
        # 对每个c值绘制一条曲线
        plt.figure()
        for c, values in data.items():
            plt.plot(values['sigma'], values['acc'], '-o', label=f"C={c}")
        plt.xlabel("Sigma")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 dataset, Gaussian kernel")
        plt.legend()
        plt.gca().set_xscale('log')
        plt.show()

    # 多项式核
    elif kernel_type[0] == 'p':
        cifar10_test_accs = {}
        for degree in [1, 2, 3, 4, 5]:
            for theta in [0, 0.5, 1, 2, 3, 4, 5]:
                print(f"CIFAR-10 dataset, Polynomial kernel, Degree={degree}, Theta={theta}")
                kernel_type = [['p', 1, degree, theta, 1]]
                cifar10_test_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False)
                cifar10_test_accs[(degree, theta)] = cifar10_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (degree, theta), acc in cifar10_test_accs.items():
            if degree not in data:
                data[degree] = {'theta': [], 'acc': []}
            data[degree]['theta'].append(theta)
            data[degree]['acc'].append(acc)
        # 对每个degree值绘制一条曲线
        plt.figure()
        for degree, values in data.items():
            plt.plot(values['theta'], values['acc'], '-o', label=f"Degree={degree}")
        plt.xlabel("Theta")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 dataset, Polynomial kernel")
        plt.legend()
        plt.show()

    # sigmoid核
    elif kernel_type[0] == 's':
        cifar10_test_accs = {}
        for beta in [1e-3, 5e-3, 1e-2, 5e-2]:
            for theta in [-3, -2, -1, 0, 1, 2, 3]:
                print(f"CIFAR-10 dataset, Sigmoid kernel, Beta={beta}, Theta={theta}")
                kernel_type = [['s', 1, beta, theta, 1]]
                cifar10_test_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False)
                cifar10_test_accs[(beta, theta)] = cifar10_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (beta, theta), acc in cifar10_test_accs.items():
            if beta not in data:
                data[beta] = {'theta': [], 'acc': []}
            data[beta]['theta'].append(theta)
            data[beta]['acc'].append(acc)
        # 对每个beta值绘制一条曲线
        plt.figure()
        for beta, values in data.items():
            plt.plot(values['theta'], values['acc'], '-o', label=f"Beta={beta}")
        plt.xlabel("Theta")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 dataset, Sigmoid kernel")
        plt.legend()
        plt.show()

    else:
        print("Invalid kernel type!")
        exit(1)

else:
    print("Invalid dataset!")
    exit(1)