from controller import rtPGM, PGM, LQR
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete

# tilting table: model as decoupled double integrators
# continuous
g = 9.81
A = np.array([[0., 1.], [0., 0.]])
B = np.array([[0.], [g]])
C = np.array([[1., 0.]])
D = np.array([[0.]])

# discrete
Ts = 0.04
A, B, C, D, _ = cont2discrete((A, B, C, D), Ts, 'zoh')

umin = -np.pi/6.
umax = np.pi/6.
nx = A.shape[0]
nq = B.shape[1]
N = 10

# controller
Q = 10000.*C.T.dot(C)
R = 0.1*np.eye(nq)
c1 = rtPGM(A, B, C, D, Q, R, umin, umax, N)
c2 = rtPGM(A, B, C, D, Q, R, umin, umax, N)
# controller = PGM(A, B, C, D, Q, R, umin, umax, N)
# controller = LQR(A, B, C, D, Q, R)
# controller.codegen()

# mpc iterates
n_it = 500
q1 = np.zeros((N, 1))
q2 = np.zeros((N, 1))
x1 = np.array([[0.1], [0.]])
x2 = np.array([[0.], [0.]])
r1 = np.c_[0].T
r2 = np.c_[0].T
x1_sv = np.zeros((nx, 0))
x2_sv = np.zeros((nx, 0))
q1_sv = np.zeros((nq, 0))
q2_sv = np.zeros((nq, 0))
r1_sv = np.zeros((1, 0))
r2_sv = np.zeros((1, 0))
VN1_sv = np.zeros((1, 0))
VN2_sv = np.zeros((1, 0))

r_freq = 0.4 # Hz

for k in range(n_it):
    x1_sv = np.hstack((x1_sv, x1))
    x2_sv = np.hstack((x2_sv, x2))
    VN1_sv = np.hstack((VN1_sv, np.c_[c1.VN(x1, r1)]))
    VN2_sv = np.hstack((VN2_sv, np.c_[c2.VN(x2, r2)]))
    # update trajectory
    r1 = np.c_[0.3*np.sin(2*np.pi*r_freq*k*Ts)]
    r2 = np.c_[0.2*np.cos(2*np.pi*r_freq*k*Ts)]
    #r1 = np.c_[0]
    #r2 = np.c_[0]
    q1 = c1.update(x1, r1)
    q2 = c2.update(x2, r2)
    r1_sv = np.hstack((r1_sv, r1))
    r2_sv = np.hstack((r2_sv, r2))
    q1_sv = np.hstack((q1_sv, q1))
    q2_sv = np.hstack((q2_sv, q2))
    # update system
    x1 = A.dot(x1) + B.dot(q1)
    x2 = A.dot(x2) + B.dot(q2)

# plot
time = np.linspace(0, (n_it-1)*Ts, n_it)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, x1_sv[0, :], 'b')
plt.plot(time, x2_sv[0, :], 'r')
plt.legend(['x1', 'x2'])
plt.plot(time, r1_sv[0, :], 'b--')
plt.plot(time, r2_sv[0, :], 'r--')
plt.ylabel('x')
plt.subplot(3, 1, 2)
plt.plot(time, q1_sv[0, :], 'b')
plt.plot(time, q2_sv[0, :], 'r')
plt.legend(['q1', 'q2'])
plt.plot([0, (n_it-1)*Ts], [umax, umax], 'r--')
plt.plot([0, (n_it-1)*Ts], [umin, umin], 'r--')
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('q')
plt.subplot(3, 1, 3)
plt.plot(time, VN1_sv[0, :], 'b')
plt.plot(time, VN2_sv[0, :], 'r')
plt.legend(['V_N1', 'V_N2'])
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('V_N')
plt.xlabel('t (s)')

plt.figure()
plt.plot(r1_sv[0, :], r2_sv[0, :], 'r--')
plt.plot(x1_sv[0, :], x2_sv[0, :], 'b')
plt.xlabel('x1')
plt.ylabel('y1')
plt.show()
