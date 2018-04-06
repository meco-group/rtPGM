from rtPGM.controller import rtPGM
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete
import csv


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


def save_simulation(filename, x_sv, q_sv, n_iter=None):
    n_samp = x_sv.shape[1]
    if n_iter is None:
        n_iter = [1 for k in range(n_samp)]
    with open(filename, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        for k in range(n_samp):
            writer.writerow([q_sv[0, k], x_sv[0, k], x_sv[1, k], x_sv[2, k], x_sv[3, k], n_iter[k]])


# system model: car with pendulum
Ts = 0.02
Ac, Bc, Cc, Dc = load_model('data/mecotron_inverted.txt')
S = np.vstack((Cc[:2, :], Cc[3:]))
A, B, CC, DD, _ = cont2discrete([Ac, Bc, Cc, Dc], Ts, method='zoh')
C = np.c_[CC[2, :]].T # pendulum position control
D = np.c_[DD[2, :]].T # pendulum position control

nx = A.shape[0]
nq = B.shape[1]
ny = C.shape[0]
umin = -0.3
umax = 0.3
N = 20

# controller
Q = CC[:2, :].T.dot(sc.linalg.block_diag(1.3e7, 100000.)).dot(CC[:2, :])
R = np.eye(nq)

# terminal controller using pole placement
wn = 2*np.pi*0.24
sig = wn*np.cos(np.pi/4.)
wd = wn*np.sin(np.pi/4.)
poles_c = np.array([-2, -wn, -sig+wd*1j, -sig-wd*1j])
poles_d = np.exp(Ts*poles_c)
fsf = sc.signal.place_poles(A, B, poles_d)
Kf = fsf.gain_matrix

# regularization on input rate
CR = np.zeros((N-1, N))
for k in range(N-1):
    CR[k, k] = 1.
    CR[k, k+1] = -1.
R_extra = 1e10*CR.T.dot(CR)
gamma_ratio = 0.8*0.99

controller = rtPGM(A, B, C, D, Q, R, umin, umax, N, Kf=Kf, R_extra=R_extra,
                   gamma_ratio=gamma_ratio, terminal_constraint=False)
# controller.codegen()

# mpc iterates
n_it = int(20./Ts)
q = np.zeros((N, 1))
x = np.linalg.inv(S).dot(np.array([[0.05, 0.0, 0.0, 0.]]).T)
r = np.array([[0.0]])

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
    x = A.dot(x) + B.dot(q)

y_sv = CC.dot(x_sv)

# plot
time = np.linspace(0, (n_it-1)*Ts, n_it)
plt.figure()
plt.subplot(5, 1, 1)
plt.plot(time, y_sv[0, :])
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('Pendulum angle')
plt.subplot(5, 1, 2)
plt.plot(time, y_sv[1, :])
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('Cart position')
plt.subplot(5, 1, 3)
plt.plot(time, y_sv[2, :])
plt.plot([0, (n_it-1)*Ts], [r[0, 0], r[0, 0]], 'r--')
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('Pendulum position')
plt.subplot(5, 1, 4)
plt.plot(time, q_sv[0, :])
plt.plot([0, (n_it-1)*Ts], [umax, umax], 'r--')
plt.plot([0, (n_it-1)*Ts], [umin, umin], 'r--')
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('q')
plt.subplot(5, 1, 5)
plt.plot(time, VN_sv[0, :])
plt.xlim([0, (n_it-1)*Ts])
plt.ylabel('V_N')
plt.xlabel('t (s)')
plt.show()

plt.show()
