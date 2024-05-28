import dataset
from SVM import SVM
import numpy as np
import itertools
from tqdm import tqdm
import time

# OVO策略
def OVO(train_data, test_data, test_labels, kernel_type, test_train=True):
    svms = {}
    start_time = time.time()
    # 对每一对类别训练一个SVM
    for pair in tqdm(itertools.combinations(range(10), 2), desc="Training SVMs", total=45):
        a, b = pair
        train_labels_a = np.ones(len(train_data[a])) * -1
        train_labels_b = np.ones(len(train_data[b]))
        train_data_ab = np.concatenate([train_data[a], train_data[b]], axis=0)
        train_labels_ab = np.concatenate([train_labels_a, train_labels_b], axis=0)

        # 训练SVM
        svm = SVM(kernel_type=kernel_type)
        svm.fit(train_data_ab, train_labels_ab)
        svms[pair] = svm
    end_time = time.time()
    print(f"Training time: {end_time - start_time}s")

    # 对训练集进行预测
    if test_train:
        train_labels = np.concatenate([np.ones(len(train_data[i])) * i for i in range(10)], axis=0)
        train_data = np.concatenate(train_data, axis=0)
        votes = np.zeros((len(train_data), 10))
        for pair, svm in tqdm(svms.items(), desc="Predicting for train data"):
            # 对每个训练样本进行预测
            predictions = svm.predict(train_data)
            # 对预测结果进行投票
            for i, prediction in enumerate(predictions):
                if prediction == -1:
                    prediction = pair[0]
                else:
                    prediction = pair[1]
                votes[i, prediction] += 1
        # 最终的预测结果是得票最多的类别
        final_predictions = np.argmax(votes, axis=1)
        correct = np.sum(final_predictions == train_labels)
        train_acc = 100 * correct / len(train_data)
        print(f"OVO Train Accuracy: {train_acc}")

    # 对测试集进行预测
    votes = np.zeros((len(test_data), 10))
    start_time = time.time()
    for pair, svm in tqdm(svms.items(), desc="Predicting for test data"):
        # 对每个测试样本进行预测
        predictions = svm.predict(test_data)
        # 对预测结果进行投票
        for i, prediction in enumerate(predictions):
            if prediction == -1:
                prediction = pair[0]
            else:
                prediction = pair[1]
            votes[i, prediction] += 1
    # 最终的预测结果是得票最多的类别
    final_predictions = np.argmax(votes, axis=1)
    correct = np.sum(final_predictions == test_labels)
    test_acc = 100 * correct / len(test_labels)
    end_time = time.time()
    print(f"OVO Test Accuracy: {test_acc}, Predicting time: {end_time - start_time}s")

    if test_train:
        return train_acc, test_acc
    else:
        return test_acc

# OVA策略
def OVA(train_data, test_data, test_labels, kernel_type, test_train=True):
    svms = []
    start_time = time.time()
    # 对每个类别训练一个SVM
    for i in tqdm(range(10), desc="Training SVMs"):
        train_data_o = train_data[i]
        train_data_not_o = np.concatenate([train_data[j] for j in range(10) if j != i], axis=0)
        train_labels_o = np.ones(len(train_data_o))
        train_labels_not_o = np.ones(len(train_data_not_o)) * -1
        train_data_all = np.concatenate([train_data_o, train_data_not_o], axis=0)
        train_labels_all = np.concatenate([train_labels_o, train_labels_not_o], axis=0)

        # 训练SVM
        svm = SVM(kernel_type=kernel_type)
        svm.fit(train_data_all, train_labels_all)
        svms.append(svm)
    end_time = time.time()
    print(f"Training time: {end_time - start_time}s")

    # 对训练集进行预测
    if test_train:
        train_labels = np.concatenate([np.ones(len(train_data[i])) * i for i in range(10)], axis=0)
        train_data = np.concatenate(train_data, axis=0)
        votes = np.zeros((len(train_data), 10))
        i = 0
        for svm in tqdm(svms, desc="Predicting for train data"):
            # 对每个训练样本进行预测
            predictions = svm.predict(train_data)
            votes[:, i] = predictions
            i += 1
        # 最终的预测结果是得票最多的类别
        final_predictions = np.argmax(votes, axis=1)
        correct = np.sum(final_predictions == train_labels)
        train_acc = 100 * correct / len(train_labels)
        print(f"OVA Train Accuracy: {train_acc}")

    # 对测试集进行预测
    votes = np.zeros((len(test_data), 10))
    i = 0
    start_time = time.time()
    for svm in tqdm(svms, desc="Predicting for test data"):
        # 对每个测试样本进行预测
        predictions = svm.predict(test_data)
        votes[:, i] = predictions
        i += 1
    # 最终的预测结果是得票最多的类别
    final_predictions = np.argmax(votes, axis=1)
    correct = np.sum(final_predictions == test_labels)
    test_acc = 100 * correct / len(test_labels)
    end_time = time.time()
    print(f"OVA Test Accuracy: {test_acc}, Predicting time: {end_time - start_time}s")

    if test_train:
        return train_acc, test_acc
    else:
        return test_acc

def use_mnist(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type, method='OVO', test_train=True):
    print("Using MNIST dataset...")
    # 展平MNIST
    mnist_train_data, mnist_test_data = dataset.flatten_dataset(mnist_train_data, mnist_test_data)

    # 训练SVM分类器完成MNIST十分类
    if method == 'OVO':
        # OVO策略
        print("Use OVO method...")
        acc = OVO(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type, test_train=test_train)
    elif method == 'OVA':
        # OVA策略
        print("Use OVA method...")
        acc = OVA(mnist_train_data, mnist_test_data, mnist_test_labels, kernel_type, test_train=test_train)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    return acc

def use_cifar10(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type, method='OVO', test_train=True):
    print("Using CIFAR-10 dataset...")
    # # 展平CIFAR-10
    # cifar10_train_data, cifar10_test_data = dataset.flatten_dataset(cifar10_train_data, cifar10_test_data)
    # 提取HOG特征
    cifar10_train_data, cifar10_test_data = dataset.extract_hog(cifar10_train_data, cifar10_test_data)

    # 训练SVM分类器完成CIFAR-10十分类
    if method == 'OVO':
        # OVO策略
        print("Use OVO method...")
        acc = OVO(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type, test_train=test_train)
    elif method == 'OVA':
        # OVA策略
        print("Use OVA method...")
        acc = OVA(cifar10_train_data, cifar10_test_data, cifar10_test_labels, kernel_type, test_train=test_train)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    return acc