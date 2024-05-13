import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

#A-Part
#1)
x = sp.symbols('x', real=True)
f = x ** 4
f_x = sp.diff(f, x)
print(f_x)

#2)
x = sp.symbols('x', real=True)
derivative_f_x = sp.lambdify(x, f_x)  # Derivative function

f = x ** 4
delta = 0.01
finite_difference = ((x + delta) ** 4 - x ** 4) / delta #Formula
finite_f_x = sp.lambdify(x, finite_difference)  # Finite difference function

X = np.arange(-150, 150, 10)
arr_1 = []
arr_2 = []

for x in X:
    arr_1.append(derivative_f_x(x))
    arr_2.append(finite_f_x(x))
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X, arr_1, label='Derivative value', color='red')
plt.plot(X, arr_2, label='Finite difference', color='black',linestyle='dotted')
plt.legend(["Deravative values",'Finite difference'],loc='upper right')
plt.grid(True)
plt.show()
#3)
x = sp.symbols('x', real=True)
derivative_f_x = sp.lambdify(x, f_x)  # Derivative function

f = x ** 4
X = np.arange(-1, 1, 0.1)

def get_function(x, delta):
    finite_difference = ((x + delta) ** 4 - x ** 4) / delta
    return sp.lambdify(x, finite_difference)

finite_f_x = get_function(x, delta=0.001)
A = get_function(x, delta=0.01)
B = get_function(x, delta=0.1)
C = get_function(x, delta=0.5)
D = get_function(x, delta=1)

arr_1 = [derivative_f_x(x_val) for x_val in X]
arr_2 = [finite_f_x(x_val) for x_val in X]
Y3 = [A(x_val) for x_val in X]
Y4 = [B(x_val) for x_val in X]
Y5 = [C(x_val) for x_val in X]
Y6 = [D(x_val) for x_val in X]

plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X, arr_1, 'x-', c='red', label='derivative value')
plt.plot(X, arr_2, 'x-', c='blue', label='δ=0.001')
plt.plot(X, Y3, 'x-', c='orange', label='δ=0.01')
plt.plot(X, Y4, 'x-', c='pink', label='δ=0.1')
plt.plot(X, Y5, 'x-', c='green', label='δ=0.5')
plt.plot(X, Y6, 'x-', c='purple', label='δ=1')
plt.legend(loc='upper right')
plt.show()

def derivative_f_x(x):
    return x ** 4

def finite_f_x(x):
    return 4*x**3

def A(x, gamma):
    return gamma*x**2

def B(x, gamma):
    return gamma*2*x

def C(x, gamma):
    return gamma*abs(x)

def D(x, gamma):
    if x< 0:
        return -gamma
    return gamma
#B-Part
#1)
def gradient_descent(derivative_f_x, finite_f_x, x0, alpha=0.15, num_iters=50):
    x = x0
    X = np.array([x])
    F = np.array(derivative_f_x(x))
    for k in range(num_iters):
        step = alpha * finite_f_x(x)
        x = x - step
        X = np.append(X, [x], axis=0)
        F = np.append(F, derivative_f_x(x))
    return (X, F)

# b.ii
x0 = 1
alpha = 0.1
(X, F) = gradient_descent(derivative_f_x, finite_f_x, x0=x0, alpha=alpha)
xx = np.arange(-1, 1.1, 0.1)

plt.figure(figsize=(8, 6))

plt.plot(F, label='f(x)', color='red')

plt.plot(X, label='x', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Values')
plt.title('f(x) and x values Over Iterations')
plt.legend()
plt.show()


# b.iii
# Gradient Descent for Different X Values
for x in np.arange(0.5, 1, 0.1):
    x0 = x
    alpha = 0.1
    (X, F) = gradient_descent(derivative_f_x, finite_f_x, x0=x0, alpha=alpha)
    xx = np.arange(-1, 1.1, 0.1)

    plt.figure(figsize=(6, 4))
    plt.step(X, derivative_f_x(X), color='green', label='Gradient Descent Path')
    plt.plot(xx, derivative_f_x(xx), color='magenta', label='Derivative f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Gradient Descent with Starting x = {x:.1f}')
    plt.legend()
    plt.grid(True)
    plt.show()


# Gradient Descent for Different Alpha Values
for alpha in [0.01, 0.1, 0.2, 0.3]:
    (X, F) = gradient_descent(derivative_f_x, finite_f_x, x0=x0, alpha=alpha)
    xx = np.arange(-1, 1.1, 0.1)

    plt.figure(figsize=(6, 4))
    plt.step(X, derivative_f_x(X), color='cyan', label='Gradient Descent Path')
    plt.plot(xx, derivative_f_x(xx), color='orange', label='Derivative f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Gradient Descent with Alpha = {alpha}')
    plt.legend()
    plt.grid(True)
    plt.show()

#c.i
gamma= 0.1 
x=1
A_1= np.array([x])
B_1 = np. array(A(x, gamma)) 
for k in range(50):
    step = 0.1 * B(x, gamma)
    x=x-step
    X_1 = np.append(A_1, [x] , axis=0) 
    F_1=np.append(B_1, A(x,gamma))
gamma= 0.5
x=1
X_2= np.array([x])
F_2 = np.array(A(x, gamma)) 
for k in range(50):
    step= 0.1* B(x,gamma) 
    x=x-step
    X_2 = np.append(A_1, [x] , axis=0)
    F_2 = np.append(B_1, A(x, gamma))
gamma= 1 
x=1
X_3 = np.array([x])
F_3 = np.array(A(x, gamma)) 
for k in range(50):
    step=0.1* B(x,gamma) 
    x=x-step
    X_3 = np.append(A_1, [x] , axis=0)
    F_3 = np.append(B_1, A(x, gamma))
gamma = 2
x = 1
X_4 =np.array([x])
F_4 = np.array(A(x, gamma)) 
for k in range(50):
    step=0.1* B(x,gamma) 
    x=x-step
    X_4 = np.append(A_1, [x] , axis=0)
    F_4 = np.append(B_1, A(x, gamma))
xx=np.arange(-1,1.1,0.1)


plt.step(X_1, A(X_1, 0.1)) 
plt.plot(xx, A(xx, 0.1)) 
plt.step(X_2, A(X_2, 0.5 ) ) 
plt.plot(xx, A(xx, 0.5)) 
plt.step(X_3, A(X_3, 1 ) )
plt.plot(xx,A(xx,1)) 
plt.step(X_4, A(X_4, 2 ) ) 
plt.plot(xx, A(xx, 2 ) ) 
plt.xlabel('x')
plt.ylabel('f(x)') 
plt.legend(['v=0.1','v=0.5','v=1','=2'])
plt.show( )

##с.іі
gamma = 0.1 
x=1
A_1 = np.array([x])
B_1 = np.array(C(x, gamma)) 
for k in range(50):
    step=0.1* D(x,gamma) 
    x=x-step
    X_1= np.append(A_1,[ x ] ,axis=0)
    F_1 = np.append(B_1, C(x, gamma))
gamma= 0.5 
x=1
X_2 = np.array([x])
F_2 = np.array(C(x, gamma)) 
for k in range(50):
    step = 0.1 * D(x, gamma) 
    x=x-step
    X_2= np.append(A_1,[ x ] ,axis=0) 
    F_2=np.append(B_1, C(x,gamma))
gamma = 1 
x=1
X_3 = np.array([x]) 
F_3=np.array(C(x,gamma))
for k in range(50):
    step=0.1* D(x,gamma) 
    x=x-step
    X_3 = np.append(A_1, [ x ] , axis=0)
    F_3 = np.append(B_1, C(x, gamma))
gamma= 2
x=1
X_4= np.array([x])
F_4=np.array(C(x,gamma)) 
for k in range(50):
    step =0.1 * D(x, gamma) 
    x=x-step
    X_4 = np.append(A_1, [ x ] , axis=0) 
    F_4=np.append(B_1, C(x,gamma))
xx= np.arange(-1, 1.1 ,0.1) 


plt.step(X_1, C(X_1, 0.1)) 
plt.plot(xx, C(xx, 0.1)) 
plt.step(X_2, C(X_2, 0.5 ) ) 
plt.plot(xx, C(xx, 0.5)) 
plt.step(X_3, C(X_3, 1 ) )
plt.plot(xx,C(xx,1)) 
plt.step(X_4, C(X_4, 2 ) ) 
plt.plot(xx, C(xx, 2 ) ) 
plt.xlabel('x')
plt.ylabel('f(x)') 
plt.legend(['v=0.1','v=0.5','v=1','v=2'])
plt.show( )