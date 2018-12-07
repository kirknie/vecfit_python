import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


def armijo_cond(line_fct, d_line_fct, alpha, c1):
    return line_fct(alpha) <= line_fct(0) + c1 * alpha * d_line_fct(0)


def curvature_cond(d_line_fct, alpha, c2):
    return d_line_fct(alpha) >= c2 * d_line_fct(0)


def strong_curvature_cond(d_line_fct, alpha, c2):
    return abs(d_line_fct(alpha)) <= abs(c2 * d_line_fct(0))


def wolfe_cond(line_fct, d_line_fct, alpha, c1, c2):
    # 0 < c1 < c2 < 1
    cond1 = armijo_cond(line_fct, d_line_fct, alpha, c1)
    cond2 = curvature_cond(d_line_fct, alpha, c2)
    return cond1 and cond2


def line_search(line_fct, d_line_fct, alpha_max, c1, c2):
    """
    Algorithm 3.5
    :param line_fct: Line function
    :param d_line_fct: Derivative of line function
    :param alpha_max: Maximum step size
    :param c1: Armijo condition (sufficient descent condition)
    :param c2: curvature condition
    :return: The line search step size
    """
    alpha_0 = 0
    alpha_1 = 0.5 * alpha_max
    alpha_list = [alpha_0, alpha_1]
    i = 1
    alpha_old = alpha_0
    alpha_new = alpha_1
    while True:
        # y_tmp = line_fct(alpha_new)
        if not armijo_cond(line_fct, d_line_fct, alpha_new, c1) or (line_fct(alpha_new) >= line_fct(alpha_old) and i > 1):
            alpha = zoom(line_fct, d_line_fct, alpha_old, alpha_new, c1, c2)
            break
        # dy_tmp = d_line_fct(alpha_new)
        if strong_curvature_cond(d_line_fct, alpha_new, c2):
            alpha = alpha_new
            break
        if d_line_fct(alpha_new) >= 0:
            alpha = zoom(line_fct, d_line_fct, alpha_new, alpha_old, c1, c2)
            break
        alpha_old = alpha_new
        alpha_new = (alpha_new + alpha_max) / 2
        i += 1
    return alpha


def zoom(line_fct, d_line_fct, alpha_low, alpha_high, c1, c2):
    """
    Algorithm 3.6, called by line_search()
    :param line_fct: Line function
    :param d_line_fct: Derivative of line function
    :param alpha_low: Minimum step size that satisfies strong Wolfe condition
    :param alpha_high: Maximum step size that satisfies strong Wolfe condition
    :param c1: Armijo condition (sufficient descent condition)
    :param c2: curvature condition
    :return: The line search step size
    """
    while True:
        # interpolate with quadratic function by evaluating alpha_low, alpha_high, and mid point
        # y = a_0 + a_1*x + a_2*x^2
        alpha_mid = (alpha_low+alpha_high)/2
        A = np.array([[1, alpha_low, alpha_low**2],
                      [1, alpha_mid, alpha_mid**2],
                      [1, alpha_high, alpha_high**2]])
        b = np.array([line_fct(alpha_low), line_fct(alpha_mid), line_fct(alpha_high)])
        x = np.linalg.solve(A, b)
        alpha_new = -x[1]/2/x[2]
        if alpha_new > alpha_high:
            alpha_new = alpha_mid
        elif alpha_new < alpha_low:
            alpha_new = alpha_mid

        if not armijo_cond(line_fct, d_line_fct, alpha_new, c1) or line_fct(alpha_new) >= line_fct(alpha_low):
            alpha_high = alpha_new
        else:
            if strong_curvature_cond(d_line_fct, alpha_new, c2):
                alpha = alpha_new
                break
            if d_line_fct(alpha_new) * (alpha_high-alpha_low) >= 0:
                alpha_high = alpha_low
            alpha_low = alpha_new
    return alpha


def back_tracing(line_fct, d_line_fct, alpha_max, rho, c):
    """
    Algorithm 3.1 back tracing line search
    :param line_fct: Line function
    :param d_line_fct: Derivative of line function
    :param alpha_max: Maximum step size
    :param rho: Back tracing step
    :param c: Sufficient descent condition
    :return: Step size
    """
    alpha = alpha_max
    while not armijo_cond(line_fct, d_line_fct, alpha, c):
        alpha *= rho
    return alpha


def gradient_descent(fct, grad, x0, return_step=False):
    """
    Gradient descent method with line search algorithm
    :param fct: Function to minimize
    :param grad: Gradient of function
    :param x0: Initial point
    :param return_step: If False, only return end result; if True, return end result and all steps
    :return: End result x (and all steps)
    """
    x_new = x0
    x_list = [x0]
    step_size = np.inf
    precision = 1e-6
    max_iter = 10000
    i = 0
    while step_size > precision and i < max_iter:
        x_old = x_new
        p = -grad(x_old)
        y = lambda x: line_fct(fct, x_old, p, x)
        dy = lambda x: d_line_fct(grad, x_old, p, x)
        # dx = line_search(y, dy, 1e-3, 1e-4, 0.1)
        dx = back_tracing(y, dy, 1e-5, 0.5, 1e-4)
        x_new = x_old + dx * p
        step_size = np.linalg.norm(dx*p)
        x_list.append(x_new)
        i += 1
        if i % 1 == 0:
            print('Step size at {} iteration: {}'.format(i, step_size))
    x = x_new
    x_list = np.array(x_list)
    if return_step:
        return x, x_list
    else:
        return x


def line_fct(fct, x0, p, alpha):
    y = fct(x0 + alpha*p)
    return y


def d_line_fct(grad, x0, p, alpha):
    dy = np.inner(grad(x0 + alpha*p), p)
    return dy


# test functions
def rosenbrock_fct(x, a=1, b=100):
    # https://en.wikipedia.org/wiki/Rosenbrock_function
    if np.size(x, 0) % 2 == 1:
        raise ValueError('Size of x must be odd!')
    x1 = x[0::2]
    x2 = x[1::2]
    y_tmp = b * (x1**2 - x2)**2 + (x1 - a)**2
    y = np.sum(y_tmp, 0)
    return y


def grad_rosenbrock_fct(x, a=1, b=100):
    if np.size(x, 0) % 2 == 1:
        raise ValueError('Size of x must be odd!')
    x1 = x[0::2]
    x2 = x[1::2]
    dy = np.zeros(x.shape)
    dy[0::2] = 2 * b * (x1**2 - x2) * 2 * x1 + 2 * (x1 - a)
    dy[1::2] = -2 * b * (x1**2 - x2)
    return dy


def plot_fct(x_list=None):
    x1 = np.linspace(-1.5, 1.5, 100)
    x2 = np.linspace(-0.5, 1.5, 100)

    x1, x2 = np.meshgrid(x1, x2)
    x = np.concatenate([np.expand_dims(x1, 0), np.expand_dims(x2, 0)], 0)
    y = rosenbrock_fct(x)

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # plot = ax.plot_surface(x1, x2, y, cmap=plt.cm.jet, linewidth=0, antialiased=False, alpha=0.5)
    # fig.colorbar(plot)

    plot = plt.contourf(x1, x2, y)
    fig.colorbar(plot)
    if x_list is not None:
        plt.plot(x_list[:, 0], x_list[:, 1])
        plt.plot(x_list[-1, 0], x_list[-1, 1], 'x')
    plt.show()


if __name__ == '__main__':
    x0 = np.array([-1.0, 0.5])
    x, x_list = gradient_descent(rosenbrock_fct, grad_rosenbrock_fct, x0, True)
    plot_fct(x_list)



