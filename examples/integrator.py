from rtPGM.controller import rtPGM, PGM, LQR
import numpy as np
import matplotlib.pyplot as plt

# system: integrator
Ts = 0.01
A = np.array([[1.]])
B = np.array([[Ts]])
C = np.array([[1]])
D = np.array([[0]])
umin = -2.
umax = 2.
nx = A.shape[0]
nq = B.shape[1]
N = 20

# controller
Q = 10000.*np.eye(nx)
R = 1.*np.eye(nq)
controller = rtPGM(A, B, C, D, Q, R, umin, umax, N)
# controller = PGM(A, B, C, D, Q, R, umin, umax, N)
# controller = LQR(A, B, C, D, Q, R)

# mpc iterates
n_it = 500
x = np.c_[0.].T
r = np.c_[0.2].T
x_sv = np.zeros((nx, 0))
q_sv = np.zeros((nq, 0))
xN_sv = np.zeros((1, 0))
VN_sv = np.zeros((1, 0))

for k in range(n_it):
    x_sv = np.hstack((x_sv, x))
    VN_sv = np.hstack((VN_sv, np.c_[controller.VN(x, r)]))
    # update trajectory
    q = controller.update(x, r)
    # save
    xN_sv = np.hstack((xN_sv, np.c_[controller.xN]))
    q_sv = np.hstack((q_sv, q))
    # update system
    x = A.dot(x) + B.dot(q)

# plot
time = np.linspace(0, (n_it-1)*Ts, n_it)
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(time, x_sv[0, :])
plt.plot([0, (n_it-1)*Ts], [r[0, 0], r[0, 0]], 'r--')
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('x')
plt.subplot(4, 1, 2)
plt.plot(time, q_sv[0, :])
plt.plot([0, (n_it-1)*Ts], [umax, umax], 'r--')
plt.plot([0, (n_it-1)*Ts], [umin, umin], 'r--')
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('q')
plt.subplot(4, 1, 3)
plt.plot(time, VN_sv[0, :])
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('V_N')
plt.subplot(4, 1, 4)
plt.plot(time, xN_sv[0, :])
plt.xlim([0, (n_it-1)*Ts])
plt.ylim([-1, 1])
plt.ylabel('xN')
plt.xlabel('t (s)')
plt.show()
