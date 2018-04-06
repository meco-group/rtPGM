from rtPGM.controller import rtPGM
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
import csv


def get_bump_disturbance(n_samp, Ts, v):
    A, L = 0.1, 5.
    zo = np.zeros((1, n_samp))
    dzo = np.zeros((1, n_samp))
    for k in range(n_samp):
        t = k*Ts
        if t <= L/v:
            zo[0, k] = 0.5*A*(1.-np.cos(2.*np.pi*v*t/L))
            dzo[0, k] = (A*np.pi*v/L)*np.sin(2.*np.pi*v*t/L)
        else:
            zo[0, k] = 0.
    return zo, dzo


def get_road_disturbance(n_samp, Ts, v):
    with open('data/roadprofile.csv', 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        x, zo = [], []
        for row in reader:
            x.append(float(row[0]))
            zo.append(float(row[1]))
    dx = x[1] - x[0]
    dt = dx/v
    n = len(x)
    zo = np.c_[zo].T
    dzo = (zo[:, 1:] - zo[:, :-1])/dt
    dzo = np.c_[dzo, dzo[0, -1]]
    if n < n_samp:
        raise ValueError('Not enough road samples!')
    return zo[:, :n_samp], dzo[:, :n_samp]


# system model: quarter car with active suspension
Ts = 0.002
ms = 320.
ks = 18000.
cs = 1000.
mu = 40
ku = 200000.
Ac = np.array([[0., 1., 0., -1.], [-ks/ms, -cs/ms, 0., cs/ms], [0., 0., 0., 1.], [ks/mu, cs/mu, -ku/mu, -cs/mu]])
Bc = np.array([[0., 0.], [1./ms, 0.], [0., -1.], [-1./mu, 0.]])
Cc = np.array([[-ks/ms, -cs/ms, 0., cs/ms]])
Dc = np.array([[1./ms, 0.]])
A, B, C, D, _ = cont2discrete([Ac, Bc, Cc, Dc], Ts, method='zoh')
nx = A.shape[0]
nq = 1
ny = C.shape[0]
umin = -1500.
umax = 1500.
N = 200

# controller
Q = C.T.dot(C)
R = np.c_[D[:, 0]].T.dot(np.c_[D[:, 0]])
S = np.c_[D[:, 0]].T.dot(C)
controller = rtPGM(A, np.c_[B[:, 0]], C, np.c_[D[:, 0]], Q, R, umin, umax, N, S)

# mpc iterates
n_it = 750
q = np.zeros((N, 1))
# dzo = np.zeros((1, n_it))
zo, dzo = get_bump_disturbance(n_it, Ts, 60.*1.e3/3600.)
# zo, dzo = get_road_disturbance(n_it, Ts, 60.*1.e3/3600.)

x = np.c_[0., 0., 0., 0.].T

x_sv = np.zeros((nx, 0))
q_sv = np.zeros((nq, 0))
VN_sv = np.zeros((1, 0))

for k in range(n_it):
    x_sv = np.hstack((x_sv, x))
    VN_sv = np.hstack((VN_sv, np.c_[controller.VN(x)]))
    # update trajectory
    q = controller.update(x)
    # save
    q_sv = np.hstack((q_sv, q))
    # update system
    x = A.dot(x) + B.dot(np.vstack((q, dzo[0, k])))

y_sv = C.dot(x_sv) + np.c_[D[:, 0]].dot(q_sv) + np.c_[D[:, 1]].dot(dzo)

# plot
time = np.linspace(0, (n_it-1)*Ts, n_it)
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(time, x_sv[0, :])
plt.plot(time, x_sv[1, :])
plt.plot(time, x_sv[2, :])
plt.plot(time, x_sv[3, :])
plt.legend(['x1', 'x2', 'x3', 'x4'])
plt.ylabel('x')
plt.xlim([0, (n_it-1)*Ts])
plt.subplot(4, 1, 2)
plt.plot(time, y_sv[0, :])
plt.xlim([0, (n_it-1)*Ts])
plt.subplot(4, 1, 3)
plt.plot(time, q_sv[0, :])
plt.plot([0, (n_it-1)*Ts], [umin, umin], 'r--')
plt.plot([0, (n_it-1)*Ts], [umax, umax], 'r--')
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('q')
plt.subplot(4, 1, 4)
plt.plot(time, VN_sv[0, :])
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('V_N')
plt.xlabel('t (s)')
plt.show()
