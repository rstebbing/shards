# solve.py

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np

from shard import Shard

# shard_gradient
def shard_gradient(I, J, alpha, X, y, k, outside_left=True, epsilon=1e-6):
    shape = I.shape[:2]
    domain = shape[::-1]
    shard = Shard(X, k, outside_left=outside_left)

    H, dX = shard(domain, return_dX=True, epsilon=epsilon)

    d = alpha * (J - y)
    R = (I - J) + d * H[..., np.newaxis]
    J = dX[..., np.newaxis] * d
    J = J.reshape(X.size, -1).transpose()
    dX = np.dot(R.ravel(), J)
    return dX.reshape(X.shape)

# main_test_shard_gradient
def main_test_shard_gradient():
    INPUT_PATH = 'data/180,180-40,40,90,127,140,40-1,0,0-2.png'
    I = plt.imread(INPUT_PATH).astype(np.float64)
    shape = I.shape[:2]
    domain = shape[::-1]

    X = np.r_[40.0, 50, 100, 127, 140, 40].reshape(-1, 2)
    outside_left = True
    y = np.r_[1.0, 0.0, 0.0]
    k = 1.0
    alpha = 0.5

    # `J` is the initial image which the shard is blended with
    J = np.zeros_like(I)
    dX = shard_gradient(I, J, alpha, X, y, k, outside_left=outside_left)

    def apply_gradient(eta):
        X_ = X + eta * dX
        shard = Shard(X_, k, outside_left=outside_left)

        H = shard(domain)
        d = alpha * (J - y)
        R = (I - J) + d * H[..., np.newaxis]
        r = R.ravel()
        E = np.dot(r, r)
        print '%.7g -> %.7g' % (eta, E)
        return H

    etas = np.arange(20) * -0.05
    n = etas.shape[0]
    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(float(n) / rows))
    w = 1.0 / cols
    h = 1.0 / rows

    f = plt.figure()
    for i, eta in enumerate(etas):
        H = apply_gradient(eta)
        Ji = (J * (1.0 - alpha * H[..., np.newaxis]) 
              + alpha * y * H[..., np.newaxis])

        r = (rows - 1) - i / cols
        c = i % cols
        ax = f.add_axes([c * w, r * h, w, h], frameon=False)
        ax.imshow(Ji)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

if __name__ == '__main__':
    main_test_shard_gradient()
    
