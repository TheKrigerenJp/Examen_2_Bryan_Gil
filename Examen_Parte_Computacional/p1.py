







































































































































































































import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, expand, simplify, lambdify, sympify
import math


def euler(f, a, b, y0, M):
    """
    Método de Euler para EDOs de primer orden.

    Parámetros:
    - f: función f(x, y)
    - a, b: intervalo [a, b]
    - y0: condición inicial
    - M: número de puntos

    Retorna:
    - xv: array de valores x
    - yv: array de valores y
    """
    h = (b - a) / (M - 1)
    xv = np.linspace(a, b, M)
    yv = [y0]
    for k in range(M - 1):
        yk = yv[k] + h * f(xv[k], yv[k])  # fórmula de Euler
        yv.append(yk)
    return xv, np.array(yv)


def metodo_euler():
    f = lambda x, y: y - x ** 2 + 1
    a = 0
    b = 2
    y0 = 0.5
    M = 5

    xv, yv = euler(f, a, b, y0, M)

    # Solución analítica
    x_graf = np.linspace(a, b, 1000)
    y_graf = (x_graf + 1) ** 2 - 0.5 * np.exp(x_graf)

    plt.plot(x_graf, y_graf, 'b', label='Solución analítica')
    plt.stem(xv, yv, linefmt='r-', markerfmt='ro', basefmt='k-', label='Euler', use_line_collection=True)
    # Si prefieres usar líneas en lugar de stem:
    # plt.plot(xv, yv, 'r-o', label='Euler')

    plt.legend()
    plt.title('Método de Euler')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def predictor_corrector(f, a, b, y0, M):
    h = (b - a) / (M - 1)
    xv = np.linspace(a, b, M)
    yv = [y0]
    for k in range(M - 1):
        zn = yv[k] + h * f(xv[k], yv[k])  # Predictor
        yk = yv[k] + (h / 2) * (f(xv[k], yv[k]) + f(xv[k+1], zn))  # Corrector
        yv.append(yk)
    return xv, np.array(yv)

def rk3(f, a, b, y0, M):
    h = (b - a) / (M - 1)
    xv = np.linspace(a, b, M)
    yv = [y0]
    for k in range(M - 1):
        k1 = f(xv[k], yv[k])
        k2 = f(xv[k] + h / 2, yv[k] + (h / 2) * k1)
        k3 = f(xv[k] + h, yv[k] + h * (2 * k1 - 1))  # As in your Octave code
        yk = yv[k] + (h / 6) * (k1 + 4 * k2 + k3)
        yv.append(yk)
    return xv, np.array(yv)

def metodo_varios_metodos():
    f = lambda x, y: y - x**2 + 1
    a = 0
    b = 2
    y0 = 0.5
    M = 5

    xvE, yvE = euler(f, a, b, y0, M)
    xvPC, yvPC = predictor_corrector(f, a, b, y0, M)
    xvRK3, yvRK3 = rk3(f, a, b, y0, M)

    # Solución analítica
    x_graf = np.linspace(a, b, 1000)
    y_graf = (x_graf + 1)**2 - 0.5 * np.exp(x_graf)

    plt.plot(x_graf, y_graf, 'b', label='Solución analítica')

    # Aproximaciones
    plt.stem(xvE, yvE, linefmt='r-', markerfmt='ro', basefmt='k-', label='Euler', use_line_collection=True)
    plt.stem(xvPC, yvPC, linefmt='g-', markerfmt='go', basefmt='k-', label='Predictor-Corrector', use_line_collection=True)
    plt.stem(xvRK3, yvRK3, linefmt='k-', markerfmt='ko', basefmt='k-', label='Runge-Kutta 3', use_line_collection=True)

    plt.legend()
    plt.title('Comparación de métodos para EDO')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def f(x):
    return 2**x + np.cos(x) - x**2

def fd(x):
    return (2**x) * np.log(2) - np.sin(x) - 2*x

def dif_fin_atras(f, x0):
    iter_max = 1000
    tol = 1e-20
    h = 1
    yk = (f(x0 + h) - f(x0)) / h
    for k in range(1, iter_max + 1):
        h = 10**(-k)
        ykN = (f(x0 + h) - f(x0)) / h
        er = abs(ykN - yk)
        if er < tol:
            print(f"dif_fin_atras convergió en k={k}")
            return ykN
        yk = ykN
    return yk

def dif_fin_centrada(f, x0):
    iter_max = 1000
    tol = 1e-20
    h = 1
    yk = (f(x0 + h) - f(x0 - h)) / (2 * h)
    for k in range(1, iter_max + 1):
        h = 10**(-k)
        ykN = (f(x0 + h) - f(x0 - h)) / (2 * h)
        er = abs(ykN - yk)
        if er < tol:
            print(f"dif_fin_centrada convergió en k={k}")
            return ykN
        yk = ykN
    return yk

def ejemplo_aprox_deriv():
    x0 = 0.5
    fd_exact = fd(x0)
    fd_aprox1 = dif_fin_atras(f, x0)
    fd_aprox2 = dif_fin_centrada(f, x0)

    print(f"Derivada exacta: {fd_exact}")
    print(f"Aproximación finita hacia adelante: {fd_aprox1}")
    print(f"Aproximación finita centrada: {fd_aprox2}")

def Lk(xv, k):
    x = symbols('x')
    n = len(xv) - 1
    L = 1
    for j in range(n + 1):
        if j != k:
            L *= (x - xv[j]) / (xv[k] - xv[j])
    return L

def pol_lagrange(xv, yv):
    n = len(xv) - 1
    p = 0
    for k in range(n + 1):
        p += yv[k] * Lk(xv, k)
    return expand(p)

def ejemplo_lagrange():
    xv = [-2, 0, 1]
    yv = [0, 1, -1]
    p = pol_lagrange(xv, yv)
    print("Polinomio de Lagrange:")
    print(p)

def simpson(f, a, b):
    """
    Regla de Simpson simple
    """
    return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))

def simpson_compuesto(f, a, b, n):
    """
    Regla de Simpson compuesta
    """
    I = 0
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    for k in range(n):
        I += simpson(f, x[k], x[k + 1])
    return I

def simpson_iterativo(f, a, b, tol=1e-10, num_inter_max=10000000):
    """
    Regla de Simpson iterativa con tolerancia
    """
    Sk = simpson_compuesto(f, a, b, 2)
    for k in range(3, num_inter_max + 1):
        Sk_N = simpson_compuesto(f, a, b, k)
        er = abs(Sk_N - Sk)
        if er < tol:
            print(f"Convergencia alcanzada en k={k}")
            return Sk_N
        Sk = Sk_N
    return Sk  # Si no converge dentro del límite

def main_simpson():
    # Puedes definir la función simbólica como string
    x = symbols('x')
    f_expr = 2**x + math.cos(1) - x**2  # Ejemplo simbólico fijo en x
    f = lambdify(x, f_expr, modules=['numpy'])

    a, b = 0, 1
    print("Simpson simple:", simpson(f, a, b))
    print("Simpson compuesto (n=10):", simpson_compuesto(f, a, b, 10))
    print("Simpson iterativo:", simpson_iterativo(f, a, b))

def trapecio(f, a, b):
    return (f(a) + f(b)) * (b - a) / 2

def trapecio_compuesto(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    I = 0
    for k in range(n):
        I += trapecio(f, x[k], x[k + 1])
    return I

def trapecio_iterativo(f, a, b, tol=1e-16, num_inter_max=10000000):
    Sk = trapecio_compuesto(f, a, b, 2)
    for k in range(3, num_inter_max + 1):
        Sk_N = trapecio_compuesto(f, a, b, k)
        er = abs(Sk_N - Sk)
        if er < tol:
            print(f"Convergencia alcanzada en k={k}")
            return Sk_N
        Sk = Sk_N
    return Sk  # si no converge

def main_trapecio():
    # Ingresar función como string
    x = symbols('x')
    f_text = "2**x + cos(x) - x**2"
    f_expr = sympify(f_text)
    f = lambdify(x, f_expr, modules=['numpy'])

    a, b = 0, 1
    print("Trapecio simple:", trapecio(f, a, b))
    print("Trapecio compuesto (n=10):", trapecio_compuesto(f, a, b, 10))
    print("Trapecio iterativo:", trapecio_iterativo(f, a, b))

def tridiagonal(n):
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1] = 1
        A[i, i] = 4
        A[i, i + 1] = 1
    return A

def vector_w(y, h):
    n = len(y)
    w = np.zeros(n)
    for i in range(1, n - 1):
        w[i] = (6 / h**2) * (y[i - 1] - 2 * y[i] + y[i + 1])
    return w

def coef_trazador_cubico(y, h):
    n = len(y)
    A = tridiagonal(n)
    w = vector_w(y, h)
    z = np.linalg.solve(A, w)
    a = np.zeros(n - 1)
    b = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for j in range(n - 1):
        a[j] = z[j + 1] / (6 * h)
        b[j] = z[j] / (6 * h)
        c[j] = y[j + 1] / h - (h / 6) * z[j + 1]
        d[j] = y[j] / h - (h / 6) * z[j]
    return a, b, c, d

def trazador_cubico():
    x = np.array([0, 2, 4, 6])
    y = 30 * 2 ** (x / 2)
    h = 2
    a, b, c, d = coef_trazador_cubico(y, h)
    print("Coeficientes del trazador cúbico:")
    print("a =", a)
    print("b =", b)
    print("c =", c)
    print("d =", d)