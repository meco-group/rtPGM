#!/usr/bin/python

from scipy.signal import cont2discrete
import csv
import numpy as np
from rtPGM.controller import PGM
import sys
import getopt


def load_model(model):
    out = []
    with open(model, 'r') as f:
        data = csv.reader(f, delimiter=' ')
        rows = [row for row in data]
        ind = 0
        for k in range(4):
            nx, _ = int(rows[ind][0]), int(rows[ind][1])
            out.append(np.array([[float(r) for r in rw] for rw in rows[ind+1: ind+1+nx]]))
            ind = ind+nx+1
    return out[0], out[1], out[2], out[3]


if __name__ == '__main__':
    N = 20
    opts, args = getopt.getopt(sys.argv[1:], 'N:')
    for opt, arg in opts:
        if opt == '-N':
            N = int(arg)
    print 'N = %d' % N

    Ts = 0.02
    Ac, Bc, Cc, Dc = load_model('../data/mecotron.txt')
    A, B, CC, DD, _ = cont2discrete([Ac, Bc, Cc, Dc], Ts, method='zoh')
    C = np.c_[CC[2, :]].T
    D = np.c_[DD[2, :]].T
    nx = A.shape[0]
    nq = B.shape[1]
    ny = C.shape[0]
    umin = -0.3
    umax = 0.3
    rho = 1.e5
    Q = 1e2*C.T.dot(C)
    R = 1e-3*np.eye(nq)
    controller = PGM(A, B, C, D, Q, R, umin, umax, N, terminal_constraint_tol=1e-9, tol=1e-4, max_iter=200000)
    controller.codegen('src/PGM.h')
