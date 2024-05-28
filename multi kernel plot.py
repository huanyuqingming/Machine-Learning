import utils
import dataset
import matplotlib.pyplot as plt

use_dataset = 'mnist'
multi_kernel = 'ggg'

'''
线性核：['l', c, kernel_weight] —— c * x1 @ x2
高斯核：['g', c, sigma, kernel_weight] —— c * exp(- |x1 - x2| ** 2 / (2 * sigma ** 2))
多项式核：['p', c, degree, theta, kernel_weight] —— c * (x1 @ x2 + theta) ** degree
sigmoid核：['s', c, beta, theta, kernel_weight] —— c * tanh(beta * x1 @ x2 + theta)
'''

# 导入数据集
cifar10, mnist = dataset.load_dataset()
cifar10_train_data, cifar10_test_data, cifar10_test_labels = cifar10
mnist_train_data, mnist_test_data, mnist_test_labels = mnist

if use_dataset == 'mnist':
    mnist_test_accs = {}
    linear = ['l', 0.1]
    gaussian = ['g', 10, 5]
    gaussian1 = ['g', 9, 4.5]
    gaussian2 = ['g', 11, 5.5]
    polynomial = ['p', 1, 2, 2]
    sigmoid = ['s', 1, 0.01, -1]
    # 高斯核+多项式核
    if multi_kernel == 'gp':
        for gw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            gaussian.append(gw)
            polynomial.append(1 - gw)
            kernel_type = [gaussian, polynomial]
            print(f"MNIST dataset, Gaussian and Polynomial kernel, Gaussian weight={gw}, Polynomial weight={1 - gw}")
            mnist_test_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False)
            mnist_test_accs[gw] = mnist_test_acc
        # 绘制参数消融结果
        plt.figure()        
        plt.plot(mnist_test_accs.keys(), mnist_test_accs.values(), '-o', label="Test Accuracy")
        plt.xlabel("Gaussian weight")
        plt.ylabel("Accuracy")
        plt.title("MNIST dataset, Gaussian and Polynomial kernel")
        plt.legend()
        plt.show()

    # 多项式核+sigmoid核
    elif multi_kernel == 'ps':
        for pw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            polynomial.append(pw)
            sigmoid.append(1 - pw)
            kernel_type = [polynomial, sigmoid]
            print(f"MNIST dataset, Polynomial and Sigmoid kernel, Polynomial weight={pw}, Sigmoid weight={1 - pw}")
            mnist_test_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False)
            mnist_test_accs[pw] = mnist_test_acc
        # 绘制参数消融结果
        plt.figure()        
        plt.plot(mnist_test_accs.keys(), mnist_test_accs.values(), '-o', label="Test Accuracy")
        plt.xlabel("Polynomial weight")
        plt.ylabel("Accuracy")
        plt.title("MNIST dataset, Polynomial and Sigmoid kernel")
        plt.legend()
        plt.show()

    # 高斯核+sigmoid核
    elif multi_kernel == 'gs':
        for gw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            gaussian.append(gw)
            sigmoid.append(1 - gw)
            kernel_type = [gaussian, sigmoid]
            print(f"MNIST dataset, Gaussian and Sigmoid kernel, Gaussian weight={gw}, Sigmoid weight={1 - gw}")
            mnist_test_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False)
            mnist_test_accs[gw] = mnist_test_acc
        # 绘制参数消融结果
        plt.figure()        
        plt.plot(mnist_test_accs.keys(), mnist_test_accs.values(), '-o', label="Test Accuracy")
        plt.xlabel("Gaussian weight")
        plt.ylabel("Accuracy")
        plt.title("MNIST dataset, Gaussian and Sigmoid kernel")
        plt.legend()
        plt.show()

    # 高斯核+多项式核+sigmoid核
    elif multi_kernel == 'gps':
        for gw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for pw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                sw = 1 - gw - pw
                if sw <= 0:
                    continue
                gaussian.append(gw)
                polynomial.append(pw)
                sigmoid.append(sw)
                kernel_type = [gaussian, polynomial, sigmoid]
                print(f"MNIST dataset, Gaussian and Polynomial and Sigmoid kernel, Gaussian weight={gw}, Polynomial weight={pw}, Sigmoid weight={sw}")
                mnist_test_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False)
                mnist_test_accs[(gw, pw)] = mnist_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (gw, pw), acc in mnist_test_accs.items():
            if gw not in data:
                data[gw] = {'pw': [], 'acc': []}
            data[gw]['pw'].append(pw)
            data[gw]['acc'].append(acc)
        # 对每个gw值绘制一条曲线
        for gw, values in data.items():
            plt.plot(values['pw'], values['acc'], '-o', label=f"Gaussian weight={gw}")
        plt.xlabel("Polynomial weight")
        plt.ylabel("Accuracy")
        plt.title("MNIST dataset, Gaussian and Polynomial and Sigmoid kernel")
        plt.legend()
        plt.show()

    # 三个高斯核
    elif multi_kernel == 'ggg':
        for gw1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for gw2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                gw3 = 1 - gw1 - gw2
                if gw3 <= 0:
                    continue
                gaussian1.append(gw1)
                gaussian2.append(gw2)
                gaussian.append(gw3)
                kernel_type = [gaussian1, gaussian2, gaussian]
                print(f"MNIST dataset, three Gaussian kernels, weights={gw1}/{gw2}/{gw3}")
                mnist_test_acc = utils.use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type=kernel_type, test_train=False)
                mnist_test_accs[(gw1, gw2)] = mnist_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (gw1, gw2), acc in mnist_test_accs.items():
            if gw1 not in data:
                data[gw1] = {'gw2': [], 'acc': []}
            data[gw1]['gw2'].append(gw2)
            data[gw1]['acc'].append(acc)
        # 对每个gw1值绘制一条曲线
        for gw1, values in data.items():
            plt.plot(values['gw2'], values['acc'], '-o', label=f"Gaussian weight 1={gw1}")
        plt.xlabel("Gaussian weight 2")
        plt.ylabel("Accuracy")
        plt.title("MNIST dataset, three Gaussian kernels")
        plt.legend()
        plt.show()

    else:
        raise ValueError("Invalid multi-kernel type.")

elif use_dataset == 'cifar10':
    cifar10_test_accs = {}
    linear = ['l', 0.5]
    gaussian = ['g', 10, 1]
    gaussian1 = ['g', 9, 0.9]
    gaussian2 = ['g', 11, 1.1]
    polynomial = ['p', 1, 4, 0.5]
    sigmoid = ['s', 1, 0.05, -1]
    # 高斯核+多项式核
    if multi_kernel == 'gp':
        for gw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            gaussian.append(gw)
            polynomial.append(1 - gw)
            kernel_type = [gaussian, polynomial]
            print(f"CIFAR-10 dataset, Gaussian and Polynomial kernel, Gaussian weight={gw}, Polynomial weight={1 - gw}")
            cifar10_test_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False)
            cifar10_test_accs[gw] = cifar10_test_acc
        # 绘制参数消融结果
        plt.figure()        
        plt.plot(cifar10_test_accs.keys(), cifar10_test_accs.values(), '-o', label="Test Accuracy")
        plt.xlabel("Gaussian weight")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 dataset, Gaussian and Polynomial kernel")
        plt.legend()
        plt.show()

    # 多项式核+sigmoid核
    elif multi_kernel == 'ps':
        for pw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            polynomial.append(pw)
            sigmoid.append(1 - pw)
            kernel_type = [polynomial, sigmoid]
            print(f"CIFAR-10 dataset, Polynomial and Sigmoid kernel, Polynomial weight={pw}, Sigmoid weight={1 - pw}")
            cifar10_test_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False)
            cifar10_test_accs[pw] = cifar10_test_acc
        # 绘制参数消融结果
        plt.figure()
        plt.plot(cifar10_test_accs.keys(), cifar10_test_accs.values(), '-o', label="Test Accuracy")
        plt.xlabel("Polynomial weight")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 dataset, Polynomial and Sigmoid kernel")
        plt.legend()
        plt.show()

    # 高斯核+sigmoid核
    elif multi_kernel == 'gs':
        for gw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            gaussian.append(gw)
            sigmoid.append(1 - gw)
            kernel_type = [gaussian, sigmoid]
            print(f"CIFAR-10 dataset, Gaussian and Sigmoid kernel, Gaussian weight={gw}, Sigmoid weight={1 - gw}")
            cifar10_test_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False)
            cifar10_test_accs[gw] = cifar10_test_acc
        # 绘制参数消融结果
        plt.figure()        
        plt.plot(cifar10_test_accs.keys(), cifar10_test_accs.values(), '-o', label="Test Accuracy")
        plt.xlabel("Gaussian weight")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 dataset, Gaussian and Sigmoid kernel")
        plt.legend()
        plt.show()

    # 高斯核+多项式核+sigmoid核
    elif multi_kernel == 'gps':
        for gw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for pw in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                sw = 1 - gw - pw
                if sw <= 0:
                    continue
                gaussian.append(gw)
                polynomial.append(pw)
                sigmoid.append(sw)
                kernel_type = [gaussian, polynomial, sigmoid]
                print(f"CIFAR-10 dataset, Gaussian and Polynomial and Sigmoid kernel, Gaussian weight={gw}, Polynomial weight={pw}, Sigmoid weight={sw}")
                cifar10_test_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False)
                cifar10_test_accs[(gw, pw)] = cifar10_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (gw, pw), acc in cifar10_test_accs.items():
            if gw not in data:
                data[gw] = {'pw': [], 'acc': []}
            data[gw]['pw'].append(pw)
            data[gw]['acc'].append(acc)
        # 对每个gw值绘制一条曲线
        for gw, values in data.items():
            plt.plot(values['pw'], values['acc'], '-o', label=f"Gaussian weight={gw}")
        plt.xlabel("Polynomial weight")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 dataset, Gaussian and Polynomial and Sigmoid kernel")
        plt.legend()
        plt.show()

    # 三个高斯核
    elif multi_kernel == 'ggg':
        for gw1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for gw2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                gw3 = 1 - gw1 - gw2
                if gw3 <= 0:
                    continue
                gaussian1.append(gw1)
                gaussian2.append(gw2)
                gaussian.append(gw3)
                kernel_type = [gaussian1, gaussian2, gaussian]
                print(f"CIFAR-10 dataset, three Gaussian kernels, weights={gw1}/{gw2}/{gw3}")
                cifar10_test_acc = utils.use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type=kernel_type, test_train=False)
                cifar10_test_accs[(gw1, gw2)] = cifar10_test_acc
        # 绘制参数消融结果
        plt.figure()
        data = {}
        for (gw1, gw2), acc in cifar10_test_accs.items():
            if gw1 not in data:
                data[gw1] = {'gw2': [], 'acc': []}
            data[gw1]['gw2'].append(gw2)
            data[gw1]['acc'].append(acc)
        # 对每个gw1值绘制一条曲线
        for gw1, values in data.items():
            plt.plot(values['gw2'], values['acc'], '-o', label=f"Gaussian weight 1={gw1}")
        plt.xlabel("Gaussian weight 2")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 dataset, three Gaussian kernels")
        plt.legend()
        plt.show()

    else:
        raise ValueError("Invalid multi-kernel type.")


