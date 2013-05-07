# solve.py

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np

from shard import Shard

# shard_gradient
def shard_gradient(Ij, alpha, X, y, k, outside_left=True, epsilon=1e-8):
    shape = Ij.shape[1:]
    domain = shape[::-1]
    shard = Shard(X, y, k, outside_left=outside_left)

    S = shard(domain)
    DX = shard.dX(domain, epsilon=epsilon)
    Dy = shard.dy(domain)

    r = (Ij - alpha * S).ravel()
    JX = (-alpha * DX).reshape(-1, r.shape[0]).transpose()
    Jy = (-alpha * Dy).reshape(-1, r.shape[0]).transpose()

    dX = np.dot(r, JX)
    dy = np.dot(r, Jy)

    return dX.reshape(X.shape), dy

# main_test_shard_gradient
def main_test_shard_gradient():
    INPUT_PATH = 'data/180,180-40,40,90,127,140,40-1,0,0-2.png'
    I = plt.imread(INPUT_PATH).transpose(2, 0, 1).astype(np.float64)
    shape = I.shape[1:]
    domain = shape[::-1]

    X = np.r_[40.0, 50, 100, 127, 140, 40].reshape(-1, 2)
    outside_left = True
    y = np.r_[1.0, 0.0, 0.0]
    k = 1.0
    alpha = 1.0

    # `J` is the initial image which the shard is blended with
    J = np.zeros_like(I)
    Ij = I - (1.0 - alpha) * J
    dX, dy = shard_gradient(Ij, alpha, X, y, k, outside_left=outside_left)

    # NOTE `apply_gradient` is NOT using `dy`
    def apply_gradient(eta):
        X_ = X + eta * dX
        shard = Shard(X_, y, k, outside_left=outside_left)
        S = shard(domain)
        r = (Ij - alpha * S).ravel()
        print '%.7g -> %.7g' % (eta, np.sum(r * r))
        return S

    etas = np.arange(16) * -0.02
    n = etas.shape[0]
    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(float(n) / rows))
    w = 1.0 / cols
    h = 1.0 / rows

    f = plt.figure()
    for i, eta in enumerate(etas):
        D = apply_gradient(eta).transpose(1, 2, 0)

        r = (rows - 1) - i / cols
        c = i % cols
        ax = f.add_axes([c * w, r * h, w, h], frameon=False)
        ax.imshow(D)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

if __name__ == '__main__':
    main_test_shard_gradient()
    
