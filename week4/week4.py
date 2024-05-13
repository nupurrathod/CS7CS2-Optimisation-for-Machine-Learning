import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


x, y = sp.symbols('x y')
f1 = 8*(x - 3)**4 + 4*(y - 1)**2
f2 = sp.Max(x - 3, 0) + 4 * sp.Abs(y - 1)

df1_dx = sp.diff(f1, x)
df1_dy = sp.diff(f1, y)
df2_dx = sp.diff(f2, x)
df2_dy = sp.diff(f2, y)

print("df1/dx:", df1_dx)
print("df1/dy:", df1_dy)
print("df2/dx:", df2_dx)
print("df2/dy:", df2_dy)

def compute_gradients(x, y): 
    return np.array([32*x**3, 4*y - 32])

def f_new(x, y):
    return np.maximum(x - 0, 0) + 2 * np.abs(y - 8)

def f_relu(x):
    return np.maximum(0, x)

def compute_gradient_relu(x):
    return np.array([1 if x > 0 else 0])

def compute_gradients_new(x, y):
    grad_x = 1 if x > 0 else 0
    grad_y = 2 if y > 8 else (-2 if y < 8 else 0) 
    return np.array([grad_x, grad_y])

def polyak(grad, prev_param, prev_grad, alpha=0.01, beta=0.9):
    param_update = -alpha * grad + beta * (prev_param - prev_grad)
    return param_update

def rmsprop(grad, s, alpha=0.01, beta=0.9, epsilon=1e-8):
    s = beta * s + (1 - beta) * (grad ** 2)
    param_update = -alpha / (np.sqrt(s) + epsilon) * grad
    return param_update, s


def heavy_ball(grad, prev_update, alpha=0.01, beta=0.9):
    param_update = -alpha * grad + beta * prev_update
    return param_update

# 4)
# Adam Function
def adam(grad, m, v, t, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param_update = -alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return param_update, m, v

def optimize_and_plot(function, gradient_function, update_function, params, alpha, beta, beta2=None, algorithm_name='RMSProp', iterations=200):

    values = [function(*params)]
    if algorithm_name == 'Adam':
        m, v = np.zeros_like(params), np.zeros_like(params)
    elif algorithm_name == 'RMSProp':
        s = np.zeros_like(params)
    prev_update = np.zeros_like(params)
    
    for t in range(1, iterations + 1):
        grad = gradient_function(*params)
        if algorithm_name == 'RMSProp':
            param_update, s = update_function(grad, s, alpha, beta)
        elif algorithm_name == 'Heavy Ball':
            param_update = update_function(grad, prev_update, alpha, beta)
        elif algorithm_name == 'Adam':
            param_update, m, v = update_function(grad, m, v, t, alpha, beta, beta2)
        

        params += param_update
        prev_update = param_update
        values.append(function(*params))
    
    # Plotting
    plt.plot(values, label=f'{algorithm_name} α={alpha}, β={beta}')


def f(x, y):
    return 8*(x-0)**4 + 2*(y-8)**2

initial_params = [3,0]  
alphas = [0.001,0.01, 0.1,0.2,0.5, 1]  
betas = [0.25, 0.9]  

# Optimization and plot for RMSProp for function 1
plt.figure(figsize=(12, 8))
for alpha in alphas:
    for beta in betas:
        optimize_and_plot(f, compute_gradients, rmsprop, initial_params.copy(), alpha, beta, algorithm_name='RMSProp')

plt.title('RMSProp Optimization for function 1')
plt.xlabel('Iteration')
plt.ylabel('Function value')
plt.legend()
plt.show()
# Optimization and plot for RMSProp with function 2
plt.figure(figsize=(12, 8))
for alpha in alphas:
    for beta in betas:
        optimize_and_plot(f_new, compute_gradients_new, rmsprop, initial_params.copy(), alpha, beta, algorithm_name='RMSProp')

plt.title('RMSProp Optimization for function 2')
plt.xlabel('Iteration')
plt.ylabel('Function value')
plt.legend()
plt.show()



# Optimization and plot for Heavy Ball with function 1
plt.figure(figsize=(12, 8))
for alpha in alphas:
    for beta in betas:
        optimize_and_plot(f, compute_gradients, heavy_ball, initial_params.copy(), alpha, beta, algorithm_name='Heavy Ball')

plt.title('Heavy Ball Optimization for function 1')
plt.xlabel('Iteration')
plt.ylabel('Function value')
plt.legend()
plt.show()

# Optimization and plot for RMSProp with function 2
plt.figure(figsize=(12, 8))
for alpha in alphas:
    for beta in betas:
        optimize_and_plot(f_new, compute_gradients_new, heavy_ball, initial_params.copy(), alpha, beta, algorithm_name='RMSProp')

plt.title('Heavy Ball Optimization for function 2')
plt.xlabel('Iteration')
plt.ylabel('Function value')
plt.legend()
plt.show()


# Optimization and plot for Adam with function 1
plt.figure(figsize=(12, 8))
beta2 = 0.999  
for alpha in alphas:
    for beta in betas:
        optimize_and_plot(f, compute_gradients, adam, initial_params.copy(), alpha, beta, beta2=beta2, algorithm_name='Adam')

plt.title('Adam Optimization for function 1')
plt.xlabel('Iteration')
plt.ylabel('Function value')
plt.legend()
plt.show()

# Optimization and plot for Adam with function 2
plt.figure(figsize=(12, 8))
beta2 = 0.999  
for alpha in alphas:
    for beta in betas:
        optimize_and_plot(f_new, compute_gradients_new, adam, initial_params.copy(), alpha, beta, beta2=beta2, algorithm_name='Adam')

plt.title('Adam Optimization for function 2')
plt.xlabel('Iteration')
plt.ylabel('Function value')
plt.legend()
plt.show()

def f_relu(x):
    return np.maximum(x, 0)

def compute_gradient_relu(x):
    return np.heaviside(x,0)


def rmsprop_single(x, grad, s, alpha, beta, epsilon=1e-8):
    s = beta * s + (1 - beta) * grad**2
    x_update = -alpha / (np.sqrt(s) + epsilon) * grad
    return x + x_update, s

def heavy_ball_single(x, grad, prev_update, alpha, beta):
    x_update = -alpha * grad + beta * prev_update
    return x + x_update, x_update

def adam_single(x, grad, m, v, t, alpha, beta1, beta2, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    x_update = -alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return x + x_update, m, v


def optimize(x_init, grad_func, update_func, update_params, iterations=100):
    x = x_init
    history = [x]

    if update_func.__name__.startswith('adam'):
        m, v = 0, 0
    else:
        prev_update = 0
    for t in range(1, iterations + 1):
        grad = grad_func(x)
        if update_func.__name__.startswith('adam'):
            x, m, v = update_func(x, grad, m, v, t, *update_params)
        else:
            x, prev_update = update_func(x, grad, prev_update, *update_params)
        history.append(x)
    return history
# Parameters
iterations = 100
x_inits = [-1, 1, 100]
alpha = 0.1
alphaadam = 0.01
alphaheavy = 1
betaheavy = 0.25
beta = 0.9
beta2 = 0.99

for x_init in x_inits:
    plt.figure(figsize=(12, 6))
    
    # RMSProp
    history = optimize(x_init, compute_gradient_relu, rmsprop_single, [alpha, beta], iterations)
    plt.plot(history, label='RMSProp')
    
    # Heavy Ball 
    history = optimize(x_init, compute_gradient_relu, heavy_ball_single, [alphaheavy, betaheavy], iterations)
    plt.plot(history, label='Heavy Ball')
    
    # Adam
    history = optimize(x_init, compute_gradient_relu, adam_single, [alphaadam, beta, beta2], iterations)
    plt.plot(history, label='Adam')
    
    plt.title(f'Optimization Trajectory for Initial x = {x_init}')
    plt.xlabel('Iteration')
    plt.ylabel('x Value')
    plt.legend()
    plt.show()