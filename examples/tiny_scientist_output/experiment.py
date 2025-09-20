import argparse
import os
import numpy as np
import json
import time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    return parser.parse_args()

def synthetic_quadratic(n=100):
    A = np.random.rand(n, n)
    A = np.dot(A, A.T)  # Make it positive definite
    b = np.random.rand(n)
    return A, b

def gradient_descent(A, b, x0=None, tol=1e-6, max_iter=1000):
    x = x0 if x0 is not None else np.zeros_like(b)
    beta = 0.5
    sigma = 1e-4
    for _ in range(max_iter):
        grad = np.dot(A, x) - b
        if np.linalg.norm(grad) < tol:
            break
        t = 1.0
        while np.linalg.norm(np.dot(A, x - t*grad) - b) > (1 - sigma * t) * np.linalg.norm(grad)**2:
            t *= beta
        x -= t * grad
    return x

def nesterovs_accelerated_gradient(A, b, x0=None, tol=1e-6, max_iter=1000):
    x = x0 if x0 is not None else np.zeros_like(b)
    y = np.copy(x)
    alpha = 0.9
    for _ in range(max_iter):
        grad = np.dot(A, y) - b
        if np.linalg.norm(grad) < tol:
            break
        x_new = y - 0.1 * grad
        y = x_new + alpha * (x_new - x)
        x = x_new
    return x

def newtons_method(A, b, x0=None, tol=1e-6, max_iter=1000):
    x = x0 if x0 is not None else np.zeros_like(b)
    for _ in range(max_iter):
        grad = np.dot(A, x) - b
        if np.linalg.norm(grad) < tol:
            break
        H = A  # Hessian is constant as A
        delta = np.linalg.solve(H, -grad)
        x += delta
    return x

def rmsprop(A, b, x0=None, tol=1e-6, max_iter=1000, lr=0.01):
    x = x0 if x0 is not None else np.zeros_like(b)
    cache = np.zeros_like(b)
    decay_rate = 0.9
    epsilon = 1e-8
    for _ in range(max_iter):
        grad = np.dot(A, x) - b
        if np.linalg.norm(grad) < tol:
            break
        cache = decay_rate * cache + (1 - decay_rate) * grad**2
        x -= lr * grad / (np.sqrt(cache) + epsilon)
    return x

def evaluate(A, b, x):
    loss = 0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x)
    return loss

def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    A, b = synthetic_quadratic(n=100)
    x0 = np.zeros_like(b)

    results = {}

    start_time = time.time()
    x_gd = gradient_descent(A, b, x0)
    results['gradient_descent'] = {
        'convergence_rate': evaluate(A, b, x_gd),
        'final_solution_accuracy': np.linalg.norm(np.dot(A, x_gd) - b),
        'computational_cost': time.time() - start_time
    }

    start_time = time.time()
    x_nag = nesterovs_accelerated_gradient(A, b, x0)
    results['nesterovs_accelerated_gradient'] = {
        'convergence_rate': evaluate(A, b, x_nag),
        'final_solution_accuracy': np.linalg.norm(np.dot(A, x_nag) - b),
        'computational_cost': time.time() - start_time
    }

    start_time = time.time()
    x_newton = newtons_method(A, b, x0)
    results['newtons_method'] = {
        'convergence_rate': evaluate(A, b, x_newton),
        'final_solution_accuracy': np.linalg.norm(np.dot(A, x_newton) - b),
        'computational_cost': time.time() - start_time
    }

    start_time = time.time()
    x_rmsprop = rmsprop(A, b, x0)
    results['rmsprop'] = {
        'convergence_rate': evaluate(A, b, x_rmsprop),
        'final_solution_accuracy': np.linalg.norm(np.dot(A, x_rmsprop) - b),
        'computational_cost': time.time() - start_time
    }

    with open(os.path.join(args.out_dir, 'final_info.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()