import numpy as np
sigma = 2
sigmaO = np.diag([3,3])
mus = [np.array([2,1]),np.array([1,2]),np.array([8,3])]
m = 10
grid = [np.array([i,j], float ) for i in range(m) for j in range(m)]

def entropy_bits(sigmaE):
    k = sigmaE.shape[0]
    return 0.5 * k * np.log2(2 * np.pi * np.e) + 0.5 * np.log2(np.linalg.det(sigmaE))
def prior_entropy():
    return sum(entropy_bits(sigmaO) for _ in mus)
def H_row(mu, x):
    d = mu - x
    return (d / np.linalg.norm(d)).reshape(1,2)
def update(sigmaU, mu, x):
    H = H_row(mu, x)
    S = H @ sigmaU @ H.T + sigma
    K = sigmaU @ H.T @ np.linalg.inv(S)
    A = np.eye(2) - K @ H
    return A @ sigmaU @ A.T + sigma * (K @ K.T)
def posterior_covariance(sensors):
    covariance = []
    for mu in mus:
        S = sigmaO.copy()
        for sensor in sensors:
            S = update(S, mu, sensor)
        covariance.append(S)
    return covariance
def posterior_entropy(sensors):
    return sum(entropy_bits(S) for S in posterior_covariance(sensors))

def greedy(n):
    X = []
    for _ in range(n):
        best, best_h = None, np.inf
        for x in grid:
            if any(np.array_equal(x, y) for y in X):
                continue
            with np.errstate(all='ignore'):
                h = posterior_entropy(X + [x])
            if h < best_h - 1e-12:
                best, best_h = x, h
        X.append(best)
    return X
if __name__ == '__main__':
    print(f"Prior entropy 1 radio: {entropy_bits(sigmaO):.4f} bits")
    print(f"Prior entropy 3 radios: {prior_entropy():.4f} bits")

    x = np.array([5,5])
    covariance = posterior_covariance([x])
    print(f"Posterior entropy at [5,5]: {posterior_entropy(x):.4f} bits")
    for i in (5,10,15):
        num1 = greedy(i)
        num2 = posterior_entropy(num1)
        print(f"posterior entropy = {num2:.4f} bits, mutual information = {prior_entropy() - num2:.4f} bits")








