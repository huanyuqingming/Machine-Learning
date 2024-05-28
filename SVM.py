import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class SVM:
    def __init__(self, C=1, tol=1e-5, max_iter=100, max_passes=5, kernel_type=[['l', 1, 1]]):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.max_passes = max_passes
        self.kernel_type = kernel_type
        self.alpha = None
        self.b = 0
        self.K = None
        self.X = None
        self.y = None

    def linear_kernel(self, x1, x2, c=1):
        return c * np.dot(x1, x2.T)
    
    def polynomial_kernel(self, x1, x2, c=1, degree=3, theta=1):
        return c * (np.dot(x1, x2.T) + theta) ** degree
    
    def gaussian_kernel(self, x1, x2, c=1, sigma=0.1):
        d = np.zeros((x1.shape[0], x2.shape[0]))
        for i in range(x1.shape[0]):
            d[i, :] = np.linalg.norm(x1[i] - x2, axis=1) ** 2
        return c * np.exp(-d / (2 * sigma ** 2))
    
    def sigmoid_kernel(self, x1, x2, c=1, beta=0.1, theta=1):
        return c * np.tanh(beta * np.dot(x1, x2.T) + theta)

    
    def kernel(self, x1, x2):
        K = np.zeros((x1.shape[0], x2.shape[0]))
        for i in range(len(self.kernel_type)):
            kernel = self.kernel_type[i]
            weight = kernel[-1]
            if kernel[0] == 'l':
                K += weight * self.linear_kernel(x1, x2, kernel[1])
            elif kernel[0] == 'p':
                K += weight * self.polynomial_kernel(x1, x2, kernel[1], kernel[2], kernel[3])
            elif kernel[0] == 'g':
                K += weight * self.gaussian_kernel(x1, x2, kernel[1], kernel[2])
            elif kernel[0] == 's':
                K += weight * self.sigmoid_kernel(x1, x2, kernel[1], kernel[2], kernel[3])
            else:
                raise ValueError(f"Invalid kernel type: {kernel[0]}")
        return K
    
    def smo(self, m, E, passes, inner_iter):
        for cnt in tqdm(range(self.max_iter), leave=False):
                if passes >= self.max_passes:
                    break
                num_changed_alphas = 0

                for i in inner_iter:
                    E[i] = np.dot((self.alpha * self.y).T, self.K[i, :]) + self.b - self.y[i]    # 对每个数据点计算误差E[i]

                    # 如果alpha[i]不满足KKT条件，则尝试优化
                    if (self.y[i] * E[i] < -self.tol and self.alpha[i] < self.C) or (self.y[i] * E[i] > self.tol and self.alpha[i] > 0):
                        j = np.random.randint(0, m) # 随机选择第二个alpha[j]
                        if j == i:
                            if i != 0:
                                j = 0
                            else:
                                j = 1
                        E[j] = np.dot((self.alpha * self.y).T, self.K[j, :]) + self.b - self.y[j]
                        alpha_i_old = self.alpha[i]
                        alpha_j_old = self.alpha[j]

                        # 计算alpha[j]的上下界H, L
                        if self.y[i] != self.y[j]:
                            L = max(0, self.alpha[j] - self.alpha[i])
                            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                        else:
                            L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                            H = min(self.C, self.alpha[j] + self.alpha[i])
                        # 如果上下界相等，无法优化
                        if L == H:
                            continue
                        
                        # 计算目标函数的二阶导数eta
                        eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                        # 如果二阶导数eta>=0，无法优化
                        if eta >= 0:
                            continue

                        # 更新alpha[j]
                        self.alpha[j] -= self.y[j] * (E[i] - E[j]) / eta
                        self.alpha[j] = min(max(self.alpha[j], L), H)
                        # 如果alpha[j]变化太小，则这次优化无显著提升，放弃优化
                        if abs(self.alpha[j] - alpha_j_old) < self.tol:
                            continue
                        # 更新alpha[i]
                        self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
                        # 更新b
                        b1 = self.b - E[i] - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                        b2 = self.b - E[j] - self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] - self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
                        if 0 < self.alpha[i] < self.C:
                            self.b = b1
                        elif 0 < self.alpha[j] < self.C:
                            self.b = b2
                        else:
                            self.b = (b1 + b2) / 2
                            
                        num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    def fit(self, X, y):
        self.X = X
        self.y = y
        m = self.X.shape[0]
        self.alpha = np.zeros(m)
        passes = 0
        E = np.zeros(m)
        opt_all = False

        # 计算核函数
        self.K = self.kernel(self.X, self.X)

        # 使用SMO算法优化SVM
        if not opt_all:
            self.smo(m, E, passes, range(m))
            opt_all = True
        else:
            not_zero_alphas = np.where(self.alpha != 0)[0]
            self.smo(m, E, passes, not_zero_alphas)
            opt_all = False

    def predict(self, X):
        f = self.alpha * self.y @ self.kernel(self.X, X) + self.b
        return np.sign(f)
    

    def visualize(self, X, y, predictions):
        plt.subplots()
        # 绘制样本点
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, label="Samples")

        # 绘制预测结果
        plt.scatter(X[:, 0], X[:, 1], marker='x', c=predictions, label="Predictions")

        # 绘制决策边界
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        epsilon = 1e-7
        w = np.dot((self.alpha * self.y).T, self.X)
        x1_1 = (-w[0] * x0_1 - self.b) / (w[1] + epsilon)
        x1_2 = (-w[0] * x0_2 - self.b) / (w[1] + epsilon)
        plt.plot([x0_1, x0_2], [x1_1, x1_2], 'k', label="Decision boundary")

        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    # 生成数据点
    def generate_data():

        num_1 = 100
        num_2 = 100
        dim = 4

        center_1 = np.ones(dim)  # 中心点1
        center_1[0] = center_1[0] * -2
        center_1[1] = center_1[1] * 2
        center_2 = np.ones(dim)  # 中心点2
        center_2[0] = center_2[0] * 2
        center_2[1] = center_2[1] * -2
        cov = np.eye(dim)  # 协方差矩阵

        X_1 = np.random.multivariate_normal(center_1, cov, num_1)
        y_1 = np.ones(num_1)

        X_2 = np.random.multivariate_normal(center_2, cov, num_2)
        y_2 = -np.ones(num_2)

        X = np.vstack((X_1, X_2))
        y = np.hstack((y_1, y_2))

        return X, y

    # 生成数据
    print("Generating data...")
    X, y = generate_data()
    print("Data generated.")

    svm = SVM()
    # 训练SVM分类器
    svm.fit(X, y)


    # 预测新样本
    X, y = generate_data()
    predictions = svm.predict(X)
    correct = np.sum(predictions == y)
    print(f"Correct: {correct}")    
    print(f"Accuracy: {100 * correct / len(y)}")

    # 可视化结果
    svm.visualize(X, y, predictions)