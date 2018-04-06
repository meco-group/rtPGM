from rtPGM.controller import rtPGM, PGM, LQR
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete

# system: mass spring damper
k = 1.e4
c = 1.e2
m = 10.

Ac = np.array([[0., 1.], [-k/m, -c/m]])
Bc = np.array([[0.], [1./m]])
Cc = np.array([[1., 0.]])
Dc = np.array([[0.]])
Ts = 0.02

A, B, C, D, _ = cont2discrete([Ac, Bc, Cc, Dc], Ts, method='zoh')

umin = -300.
umax = 300.
nx = A.shape[0]
nq = B.shape[1]
N = 8

# controller
Q = 1e5*np.eye(nx)
qi = 0.001
R = 1.*np.eye(nq)
controller = rtPGM(A, B, C, D, Q, R, umin, umax, N, integral_fb=True, integral_weight=qi)
# controller = PGM(A, B, C, D, Q, R, umin, umax, N, integral_fb=True, integral_weight=qi)
# controller = LQR(A, B, C, D, Q, R, integral_fb=True, integral_weight=1e6)
# controller.codegen()

# mpc iterates
n_it = 500
x = np.c_[0.1, 0.].T
r = np.c_[0.].T
x_sv = np.zeros((nx, 0))
q_sv = np.zeros((nq, 0))
VN_sv = np.zeros((1, 0))

for k in range(n_it):
    x_sv = np.hstack((x_sv, x))
    VN_sv = np.hstack((VN_sv, np.c_[controller.VN(x, r)]))
    # update trajectory
    q = controller.update(x, r)
    # save
    q_sv = np.hstack((q_sv, q))
    # update system
    q += 100 # test integral action
    x = A.dot(x) + B.dot(q)

# plot
time = np.linspace(0, (n_it-1)*Ts, n_it)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, x_sv[0, :])
plt.plot([0, (n_it-1)*Ts], [r[0, 0], r[0, 0]], 'r--')
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('x')
plt.subplot(3, 1, 2)
plt.plot(time, q_sv[0, :])
plt.plot([0, (n_it-1)*Ts], [umax, umax], 'r--')
plt.plot([0, (n_it-1)*Ts], [umin, umin], 'r--')
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('q')
plt.subplot(3, 1, 3)
plt.plot(time, VN_sv[0, :])
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('V_N')
plt.xlabel('t (s)')
plt.show()
