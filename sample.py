# sample.py

# Imports
import numpy as np

# sample_polygon
def sample_polygon(n):
    t = np.random.uniform(0.0, 2.0 * np.pi, n)
    t = np.sort(t)
    r = np.random.uniform(0.0, 1.0)
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    b = np.amax(X, axis=0) - np.amin(X, axis=0)
    X /= np.amax(b)
    X -= np.mean(X, axis=0)
    return X

# main_test_sample_polygon
def main_test_sample_polygon():
    import matplotlib.pyplot as plt
    from matplotlib import cm

    N = 10
    n = 4

    cols = cm.jet(np.linspace(0.0, 1.0, N, endpoint=True))

    f, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    for i in xrange(N):
        X = sample_polygon(n)
        x, y = np.transpose(np.r_['0,2', X, X[0]])
        ax.plot(x, y, '.-', c=cols[i])

    plt.show()

if __name__ == '__main__':
    main_test_sample_polygon()

