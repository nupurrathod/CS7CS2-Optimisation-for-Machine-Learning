import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from enum import Enum
import sympy as sp
"""Function for the Assignment"""
def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        z=x-w-1
        y=y+min(16*(z[0]**2+z[1]**2), (z[0]+5)**2+(z[1]+10)**2)   
        count=count+1
    return y/count

TD = generate_trainingdata()
x = np.linspace(start=-20, stop=20, num=100)
y = np.linspace(start=-20, stop=20, num=100)
z = []
for i in x:
    k = []
    for j in y:
        params = [i, j]
        k.append(f(params, TD))
    z.append(k)
z = np.array(z)
x, y = np.meshgrid(x, y)
contour_ax = plt.subplot(111, projection='3d')
contour_ax.contour3D(x, y, z, 60)
contour_ax.set_xlabel('Parameter X1')
contour_ax.set_ylabel('Parameter X2')
contour_ax.set_zlabel('F(x, T)')
contour_ax.set_title('Contour plot for function')
wireFrame_ax = plt.subplot(111, projection='3d')
wireFrame_ax.plot_wireframe(x, y, z, rstride=15, cstride=10)
wireFrame_ax.set_xlabel('Parameter X1')
wireFrame_ax.set_ylabel('Parameter X2')
wireFrame_ax.set_zlabel('F(x, T)')
wireFrame_ax.set_title('WireFrame plot for function')
plt.show()

#A(iii)
x0, x1, w0, w1 = sp.symbols('x0 x1 w0 w1', real=True)
f = sp.Min(16 * ((x0 - w0)**2 + (x1 - w1)**2), ((x0 - w0 + 5)**2 + (x1 - w1 + 10)**2))
df_one = sp.diff(f, x0)
df_two = sp.diff(f, x1)
print(df_one)
print(df_two)

#b(i)
def generate_trainingdata(m=25):
    return np.array([0,0])+0.25*np.random.randn(m,2)

def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y=0; count=0
    for w in minibatch:
        z=x-w-1
        y=y+min(16*(z[0]**2+z[1]**2), (z[0]+5)**2+(z[1]+10)**2)   
        count=count+1
    return y/count

"""Stochastic_Gradient_Descent Algorithm Implementation"""
class ALGO(Enum):
    constant_algo = 0
    polyak_algo = 1
    rmsprop_algo = 2
    heavyball_algo = 3
    adam_algo = 4

class Stochastic_Gradient_Descent:
    def __init__(self, f, df, x, algorithm, params, batch_size, training_data):
        self.epsilon = 1e-8
        self.f = f
        self.df = df
        self.x = deepcopy(x)
        self.n = len(x)
        self.params = params
        self.batch_size = batch_size
        self.training_data = training_data
        self.records = {
            'x': [deepcopy(self.x)],
            'f': [self.f(self.x, self.training_data)],
            'step': []
        }
        self.algorithm = self.get_algorithm(algorithm)
        self.initial(algorithm)

    def minibatch(self):
        # shuffle training data
        np.random.shuffle(self.training_data)
        n = len(self.training_data)
        for i in range(0, n, self.batch_size):
            if i + self.batch_size > n:
                continue
            data = self.training_data[i:(i + self.batch_size)]
            self.algorithm(data)
        self.records['x'].append(deepcopy(self.x))
        self.records['f'].append(self.f(self.x, self.training_data))

    def get_algorithm(self, algorithm):
        if algorithm == ALGO.constant_algo:
            return self.constant_algo
        elif algorithm == ALGO.polyak_algo:
            return self.polyak_algo
        elif algorithm == ALGO.rmsprop_algo:
            return self.rmsprop_algo
        elif algorithm == ALGO.heavyball_algo:
            return self.heavy_ball_algo
        else:
            return self.adam_algo

    def initial(self, algorithm):
        if algorithm == ALGO.rmsprop_algo:
            self.records['step'] = [[self.params['alpha']] * self.n]
            self.vars = {
                'sums': [0] * self.n,
                'alphas': [self.params['alpha']] * self.n
            }
        elif algorithm == ALGO.heavyball_algo:
            self.records['step'] = [0]
            self.vars = {
                'z': 0
            }
        elif algorithm == ALGO.adam_algo:
            self.records['step'] = [[0] * self.n]
            self.vars = {
                'ms': [0] * self.n,
                'vs': [0] * self.n,
                'step': [0] * self.n,
                't': 0
            }

    def constant_algo(self, data):
        alpha = self.params['alpha']
        for i in range(self.n):
            self.x[i] -= alpha * self.derivative(i, data)
        self.records['step'].append(alpha)

    def polyak_algo(self, data):
        Sum = 0
        for i in range(self.n):
            Sum = Sum + self.derivative(i, data) ** 2
        step = self.f(self.x, data) / (Sum + self.epsilon)
        for i in range(self.n):
            self.x[i] -= step * self.derivative(i, data)
        self.records['step'].append(step)

    def rmsprop_algo(self, data):
        alpha = self.params['alpha']
        beta = self.params['beta']
        alphas = self.vars['alphas']
        sums = self.vars['sums']
        for i in range(self.n):
            self.x[i] -= alphas[i] * self.derivative(i, data)
            sums[i] = (beta * sums[i]) + ((1 - beta) * (self.derivative(i, data) ** 2))
            alphas[i] = alpha / ((sums[i] ** 0.5) + self.epsilon)
        self.records['step'].append(deepcopy(alphas))

    def heavy_ball_algo(self, data):
        alpha = self.params['alpha']
        beta = self.params['beta']
        z = self.vars['z']
        Sum = 0
        for i in range(self.n):
            Sum += self.derivative(i, data) ** 2

        z = (beta * z) + (alpha * self.f(self.x, data) / (Sum + self.epsilon))
        for i in range(self.n):
            self.x[i] -= z * self.derivative(i, data)
        self.vars['z'] = z
        self.records['step'].append(z)

    def adam_algo(self, data):
        alpha = self.params['alpha']
        beta1 = self.params['beta1']
        beta2 = self.params['beta2']
        ms = self.vars['ms']
        vs = self.vars['vs']
        step = self.vars['step']
        t = self.vars['t']
        t += 1
        for i in range(self.n):
            ms[i] = (beta1 * ms[i]) + ((1 - beta1)*self.derivative(i, data))
            vs[i] = (beta2 * vs[i]) + ((1 - beta2)*(self.derivative(i, data) ** 2))
            _m = ms[i] / (1 - (beta1 ** t))
            _v = vs[i] / (1 - (beta2 ** t))
            step[i] = alpha * (_m / ((_v ** 0.5) + self.epsilon))
            self.x[i] -= step[i]
        self.vars['t'] = t
        self.records['step'].append(deepcopy(step))

    def derivative(self, i, data):
        Sum = 0
        for j in range(self.batch_size):
            Sum = Sum + self.df[i](*self.x, *data[j])
        return Sum / self.batch_size

def df_one(x0, x1, w0, w1):
    heaviside_1 = np.heaviside(-16 * (-w0 + x0 - 1)**2 + (-w0 + x0 + 5)**2 - 16 * (-w1 + x1 - 1)**2 + (-w1 + x1 + 10)**2, 0)
    heaviside_2 = np.heaviside(16 * (-w0 + x0 - 1)**2 - (-w0 + x0 + 5)**2 + 16 * (-w1 + x1 - 1)**2 - (-w1 + x1 + 10)**2, 0)
    
    term1 = (-32 * w0 + 32 * x0 - 32) * heaviside_1
    term2 = (-2 * w0 + 2 * x0 + 10) * heaviside_2
    
    return term1 + term2

def df_two(x0, x1, w0, w1):
    heaviside_1 = np.heaviside(-16 * (-w0 + x0 - 1)**2 + (-w0 + x0 + 5)**2 - 16 * (-w1 + x1 - 1)**2 + (-w1 + x1 + 10)**2, 0)
    heaviside_2 = np.heaviside(16 * (-w0 + x0 - 1)**2 - (-w0 + x0 + 5)**2 + 16 * (-w1 + x1 - 1)**2 - (-w1 + x1 + 10)**2, 0)
    
    term1 = (-32 * w1 + 32 * x1 - 32) * heaviside_1
    term2 = (-2 * w1 + 2 * x1 + 20) * heaviside_2
    
    return term1 + term2
    

colors = ['dodgerblue', 'limegreen', 'tomato', 'orchid', 'teal']


def plots(f, training_data, xs, legend):
    X_Data = np.linspace(-20, 20, 100)
    Y_Data = np.linspace(-20, 20, 100)
    Z_Data = []
    for x in X_Data:
        z = []
        for y in Y_Data: z.append(f([x, y], training_data))
        Z_Data.append(z)
    Z = np.array(Z_Data)
    X, Y = np.meshgrid(X_Data, Y_Data)
    plt.contour(X, Y, Z, 60)
    plt.xlabel('x0')
    plt.ylabel('x1')
    for i in range(len(xs)):
        x0 = [point[0] for point in xs[i]]  
        x1 = [point[1] for point in xs[i]] 
        plt.plot(x0, x1, color=colors[i % len(colors)], marker='*', markeredgecolor=colors[i % len(colors)], markersize=3)
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
    plt.legend(legend)
    plt.show()
"""Question B"""
#b(1)
def plot_sgd_performance1(f, df):
    training_data = generate_trainingdata()
    count = 100
    iters = list(range(count + 1))
    step_sizes = [0.1, 0.01, 0.001, 0.0001]
    labels = [f'step size = ${step_size}$' for step_size in step_sizes]
    xs = []
    fs = []
    for i, step_size in enumerate(step_sizes):
        sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.constant_algo, {'alpha': step_size}, batch_size=len(training_data),
                  training_data=training_data)
        for _ in range(count):
            sgd.minibatch()
        plt.plot(iters, sgd.records['f'])
        xs.append(deepcopy(sgd.records['x']))
        fs.append(deepcopy(sgd.records['f']))

    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend(labels)
    plt.show()
    plots(f, training_data, xs, labels)
#b2
def plot_sgd_performance2(f, df):
    training_data = generate_trainingdata()
    times = 5
    count = 80
    iters = list(range(count + 1))
    alpha = 0.1
    labels = [f'Trails ${i + 1}$' for i in range(times)]
    xs = []
    fs = []
    for trial in range(times):
        sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.constant_algo, {'alpha': alpha}, 5, training_data)
        for _ in range(count):
            sgd.minibatch()
        plt.plot(iters, sgd.records['f'], label=labels[trial])
        xs.append(deepcopy(sgd.records['x']))
        fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 20])
    plt.xlabel('Iterations')
    plt.ylabel('function Value')
    plt.legend()
    plt.show()
    plots(f, training_data, xs, labels)
#b3
def plot_sgd_performance3(f, df):
    training_data = generate_trainingdata()
    count = 50
    iters = list(range(count + 1))
    alpha = 0.1
    batch_sizes = [1, 5, 10, 25, 30]
    labels = [f'$batch size={n}$' for n in batch_sizes]
    xs = []
    fs = []
    for i, n in enumerate(batch_sizes):
        sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.constant_algo, {'alpha': alpha}, n, training_data)
        for _ in range(count):
            sgd.minibatch()
        plt.plot(iters, sgd.records['f'], label=labels[i])
        xs.append(deepcopy(sgd.records['x']))
        fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 3])
    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    plots(f, training_data, xs, labels)
#b4
def plot_sgd_performance4(f, df):
    training_data = generate_trainingdata()
    count = 30
    iters = list(range(count + 1))
    step_sizes = [0.1, 0.01, 0.001, 0.0001]
    labels = [f'step size = ${step_size}$' for step_size in step_sizes]
    xs = []
    fs = []
    for i, step_size in enumerate(step_sizes):
        sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.constant_algo, {'alpha': step_size}, 5, training_data)
        for _ in range(count):
            sgd.minibatch()
        plt.plot(iters, sgd.records['f'], label=labels[i])
        xs.append(deepcopy(sgd.records['x']))
        fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 120])
    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    plots(f, training_data, xs, labels)

"""Question C"""
def SGD_Polyak(f, df):
    training_data = generate_trainingdata()
    count = 100
    iters = list(range(count + 1))
    xs = []
    fs = []
    labels = ['Baseline']
    Sgd_base = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.constant_algo, {'alpha': 0.1}, 5, training_data)
    for _ in range(count):
        Sgd_base.minibatch()
    plt.plot(iters, Sgd_base.records['f'], label=labels[0])
    xs.append(deepcopy(Sgd_base.records['x']))
    fs.append(deepcopy(Sgd_base.records['f']))
    batch_sizes = [1, 5, 10, 15, 20]
    for n in batch_sizes:
        sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.polyak_algo, {}, n, training_data)
        for _ in range(count):
            sgd.minibatch()
        labels.append(f'batch size=${n}$')
        plt.plot(iters, sgd.records['f'], label=labels[-1])
        xs.append(deepcopy(sgd.records['x']))
        fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 60])
    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    plots(f, training_data, xs, labels)

def SGD_RMSProp(f, df):
    count = 100
    iters = list(range(count + 1))
    training_data = generate_trainingdata()
    xs = []
    fs = []
    labels = ['Baseline']
    Sgd_base = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.constant_algo, {'alpha': 0.1}, 5, training_data)
    for _ in range(count):
        Sgd_base.minibatch()
    plt.plot(iters, Sgd_base.records['f'], label=labels[0])
    xs.append(deepcopy(Sgd_base.records['x']))
    fs.append(deepcopy(Sgd_base.records['f']))
    alphas = [0.1, 0.01, 0.001]
    betas = [0.25, 0.9]
    for alpha in alphas:
        for beta in betas:
            sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.rmsprop_algo,
                      {'alpha': alpha, 'beta': beta}, 5, training_data)
            for _ in range(count):
                sgd.minibatch()
            labels.append(f'$\\alpha={alpha},\\,\\beta={beta}$')
            plt.plot(iters, sgd.records['f'], label=labels[-1])
            xs.append(deepcopy(sgd.records['x']))
            fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 60])
    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    xs = []
    fs = []
    labels = ['Baseline']
    plt.plot(iters, Sgd_base.records['f'], label=labels[0])
    xs.append(deepcopy(Sgd_base.records['x']))
    fs.append(deepcopy(Sgd_base.records['f']))
    batch_sizes = [1, 5, 10, 15, 20]
    for batch_size in batch_sizes:
        sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.rmsprop_algo, {'alpha': 0.1, 'beta': 0.9}, 5, training_data)
        for _ in range(count):
            sgd.minibatch()
        labels.append(f'batch size=${batch_size}$')
        plt.plot(iters, sgd.records['f'], label=labels[-1])
        xs.append(deepcopy(sgd.records['x']))
        fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 10])
    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    plots(f, training_data, xs, labels)

def SGD_HeavyBall(f, df):
    training_data = generate_trainingdata()
    count = 100
    iters = list(range(count + 1))
    xs = []
    fs = []
    labels = ['Baseline']
    Sgd_base = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.constant_algo, {'alpha': 0.1}, 5, training_data)
    for _ in range(count):
        Sgd_base.minibatch()
    plt.plot(iters, Sgd_base.records['f'], label=labels[0])
    xs.append(deepcopy(Sgd_base.records['x']))
    fs.append(deepcopy(Sgd_base.records['f']))
    alphas = [0.1, 0.01, 0.001]
    betas = [0.25, 0.9]
    for alpha in alphas:
        for beta in betas:
            sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.heavyball_algo,
                      {'alpha': alpha, 'beta': beta}, 5, training_data)
            for _ in range(count):
                sgd.minibatch()
            labels.append(f'$\\alpha={alpha},\\,\\beta={beta}$')
            plt.plot(iters, sgd.records['f'], label=labels[-1])
            xs.append(deepcopy(sgd.records['x']))
            fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 60])
    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    xs, fs = [], []
    labels = ['Baseline']
    plt.plot(iters, Sgd_base.records['f'], label=labels[0])
    xs.append(deepcopy(Sgd_base.records['x']))
    fs.append(deepcopy(Sgd_base.records['f']))
    batch_sizes = [1, 5, 10, 15, 20]
    for batch_size in batch_sizes:
        sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.heavyball_algo, {'alpha': 0.1, 'beta': 0.25}, 5, training_data)
        for _ in range(count):
            sgd.minibatch()
        labels.append(f'batch size=${batch_size}$')
        plt.plot(iters, sgd.records['f'], label=labels[-1])
        xs.append(deepcopy(sgd.records['x']))
        fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 10])
    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    plots(f, training_data, xs, labels)

def SGD_Adam(f, df):
    training_data = generate_trainingdata()
    count = 100
    iters = list(range(count + 1))
    xs = []
    fs = []
    labels = ['Baseline']
    Sgd_base = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.constant_algo, {'alpha': 0.1}, 5, training_data)
    for _ in range(count):
        Sgd_base.minibatch()
    plt.plot(iters, Sgd_base.records['f'], label=labels[0])
    xs.append(deepcopy(Sgd_base.records['x']))
    fs.append(deepcopy(Sgd_base.records['f']))
    alphas = [10, 1, 0.1]
    beta1s = [0.25, 0.9]
    beta2s = [0.999]
    for alpha in alphas:
        for beta1 in beta1s:
            for beta2 in beta2s:
                sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.adam_algo,
                          {'alpha': alpha, 'beta1': beta1, 'beta2': beta2}, 5, training_data)
                for _ in range(count):
                    sgd.minibatch()
                labels.append(
                    f'$\\alpha={alpha},\\,\\beta_1={beta1},\\,\\beta_2={beta2}$'
                )
                plt.plot(iters, sgd.records['f'], label=labels[-1])
                xs.append(deepcopy(sgd.records['x']))
                fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 60])
    plt.xlabel('iterations')
    plt.ylabel('f')
    plt.legend()
    plt.show()
    xs = []
    fs = []
    labels = ['Baseline']
    plt.plot(iters, Sgd_base.records['f'], label=labels[0])
    xs.append(deepcopy(Sgd_base.records['x']))
    fs.append(deepcopy(Sgd_base.records['f']))
    batch_sizes = [1, 3, 5, 10, 25]
    for batch_size in batch_sizes:
        sgd = Stochastic_Gradient_Descent(f, df, [3, 3], ALGO.adam_algo,
                  {'alpha': 10, 'beta1': 0.9, 'beta2': 0.999}, 5, training_data)
        for _ in range(count):
            sgd.minibatch()
        labels.append(f'batch size=${batch_size}$')
        plt.plot(iters, sgd.records['f'], label=labels[-1])
        xs.append(deepcopy(sgd.records['x']))
        fs.append(deepcopy(sgd.records['f']))
    plt.ylim([0, 10])
    plt.xlabel('iterations')
    plt.ylabel('$f(x, T)$')
    plt.legend()
    plt.show()
    plots(f, training_data, xs, labels)
    

if __name__ == '__main__':
    plot_sgd_performance1(f, [df_one, df_two])
    plot_sgd_performance2(f, [df_one, df_two])
    plot_sgd_performance3(f, [df_one, df_two])
    plot_sgd_performance4(f, [df_one, df_two])

    SGD_Polyak(f, [df_one, df_two])
    SGD_RMSProp(f, [df_one, df_two])
    SGD_HeavyBall(f, [df_one, df_two])
    SGD_Adam(f, [df_one, df_two])
