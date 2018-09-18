#!/usr/bin/python

import matplotlib.pyplot as plt
import subprocess
import csv
import numpy as np
from mecotron_codegen import nonlinear_codegen, load_model, lti_codegen
from scipy.signal import cont2discrete
from rtPGM.controller import rtPGM, PGM


def get_dict(file):
    values = []
    header = []
    with open(file, mode='r') as f:
        reader = csv.reader(f)
        for k, row in enumerate(reader):
            if k == 0:
                for key in row:
                    header.append(key)
                    values.append([])
            else:
                for l, v in enumerate(row):
                    values[l].append(float(v))
    ret = {h: np.array(v) for h, v in zip(header, values)}
    return ret

approaches = [False, True, True]

n_steps = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# default model
Ts = 0.02
Ac, Bc, Cc, Dc = load_model('../data/mecotron.txt')
A1, B1, CC, DD, _ = cont2discrete([Ac, Bc, Cc, Dc], Ts, method='zoh')
C1 = np.c_[CC[2, :]].T
D1 = np.c_[DD[2, :]].T
nx = A1.shape[0]
nq = B1.shape[1]
ny = C1.shape[0]
umin = -0.3
umax = 0.3
N = 20
rho = 1.e5
r = 0.12
Q = rho*C1.T.dot(C1)
R = np.eye(nq)
Tfinv = np.vstack((CC[1, :], CC[4, :], CC[0, :], CC[3, :]))
Tf = np.linalg.inv(Tfinv)
g = 9.81
c = 0.1955
l = 0.13
tau = 0.1
n_it = 100

# reference objective
controller = rtPGM(A1, B1, C1, D1, Q, R, umin, umax, N, terminal_constraint_tol=1e-8)
controller.codegen('src/rtPGM.h', time_analysis=True)
nonlinear_codegen(g, c, l, tau, Ts, Ts/10, Tf, CC.dot(Tf), DD, 'mecotron_nl', 'src/mecotron_nl.h')
lti_codegen(A1, B1, CC, DD, Q, R, [umin], [umax], controller.Tx, controller.Su.T, Ts, 'mecotron', 'src/mecotron.h')
subprocess.check_output("cd build && make && cd ..", shell=True)
subprocess.check_output("./build/mecotron_step_rtpgm -v 1 -t 1 -f %s -n 1 -i %d" % ('mecotron_step_rtpgm.csv', n_it), shell=True, stderr=subprocess.STDOUT)
ref = get_dict('mecotron_step_rtpgm.csv')
obj = sum(rho*(r - ref['pendulum_position'])**2 + ref['u']**2)/n_it

for ns in n_steps:
    print '\nn_it = %d' % ns
    print '--------'
    if approaches[0]:
        # 1 pgm step in Ts/ns with horizon N
        Ts2 = Ts/ns
        N2 = N
        n_it2 = n_it*ns
        A2, B2, CC, DD, _ = cont2discrete([Ac, Bc, Cc, Dc], Ts2, method='zoh')
        C2 = np.c_[CC[2, :]].T
        D2 = np.c_[DD[2, :]].T
        controller2 = rtPGM(A2, B2, C2, D2, Q, R, umin, umax, N2, terminal_constraint_tol=1e-8)
        controller2.codegen('src/rtPGM.h', time_analysis=True)
        nonlinear_codegen(g, c, l, tau, Ts2, Ts2/10, Tf, CC.dot(Tf), DD, 'mecotron_nl', 'src/mecotron_nl.h')
        lti_codegen(A2, B2, CC, DD, Q, R, [umin], [umax], controller2.Tx, controller2.Su.T, Ts2, 'mecotron', 'src/mecotron.h')
        subprocess.check_output("cd build && make && cd ..", shell=True)
        subprocess.check_output("./build/mecotron_step_rtpgm -v 1 -t 1 -f %s -n 1 -i %d" % ('mecotron_step_rtpgm_%dit_a.csv' % (ns), n_it2), shell=True, stderr=subprocess.STDOUT)
    if approaches[1]:
        # 1 pgm step in Ts/ns with horizon ns*N
        Ts2 = Ts/ns
        N2 = ns*N
        n_it2 = n_it*ns
        A2, B2, CC, DD, _ = cont2discrete([Ac, Bc, Cc, Dc], Ts2, method='zoh')
        C2 = np.c_[CC[2, :]].T
        D2 = np.c_[DD[2, :]].T
        controller2 = rtPGM(A2, B2, C2, D2, Q, R, umin, umax, N2, terminal_constraint_tol=1e-8)
        controller2.codegen('src/rtPGM.h', time_analysis=True)
        nonlinear_codegen(g, c, l, tau, Ts2, Ts2/10, Tf, CC.dot(Tf), DD, 'mecotron_nl', 'src/mecotron_nl.h')
        lti_codegen(A2, B2, CC, DD, Q, R, [umin], [umax], controller2.Tx, controller2.Su.T, Ts2, 'mecotron', 'src/mecotron.h')
        subprocess.check_output("cd build && make && cd ..", shell=True)
        subprocess.check_output("./build/mecotron_step_rtpgm -v 1 -t 1 -f %s -n 1 -i %d" % ('mecotron_step_rtpgm_%dit_b.csv' % (ns), n_it2), shell=True, stderr=subprocess.STDOUT)
    if approaches[2]:
        # ns pgm steps in Ts with horizon N
        Ts2 = Ts
        N2 = N
        n_it2 = n_it
        controller2 = PGM(A1, B1, C1, D1, Q, R, umin, umax, N2, terminal_constraint_tol=1e-8, max_iter=ns, tol=1e-16)
        controller2.codegen('src/PGM.h')
        nonlinear_codegen(g, c, l, tau, Ts2, Ts2/10, Tf, CC.dot(Tf), DD, 'mecotron_nl', 'src/mecotron_nl.h')
        lti_codegen(A2, B2, CC, DD, Q, R, [umin], [umax], controller2.Tx, controller2.Su.T, Ts2, 'mecotron', 'src/mecotron.h')
        subprocess.check_output("cd build && make && cd ..", shell=True)
        subprocess.check_output("./build/mecotron_step_pgm -v 1 -t 1 -f %s -n 1 -i %d" % ('mecotron_step_pgm_%dit.csv' % (ns), n_it2), shell=True, stderr=subprocess.STDOUT)

data = {}
for ns in n_steps:
    data[ns] = {}
    for k in range(3):
        if approaches[0]:
            data[ns][0] = get_dict('mecotron_step_rtpgm_%dit_a.csv' % (ns))
        if approaches[1]:
            data[ns][1] = get_dict('mecotron_step_rtpgm_%dit_b.csv' % (ns))
        if approaches[2]:
            data[ns][2] = get_dict('mecotron_step_pgm_%dit.csv' % (ns))

n_iter = data.keys()
obj_rel = [[] for _ in range(3)]

for ns, d in data.items():
    if approaches[0]:
        obj0 = sum(rho*(r - d[0]['pendulum_position'])**2 + d[0]['u']**2)/(n_it*ns)
        obj_rel[0].append((obj0 - obj)*100/obj0)
    else:
        obj_rel[0].append(0)
    if approaches[1]:
        obj1 = sum(rho*(r - d[1]['pendulum_position'])**2 + d[1]['u']**2)/(n_it*ns)
        obj_rel[1].append((obj1 - obj)*100/obj1)
    else:
        obj_rel[1].append(0)
    if approaches[2]:
        obj2 = sum(rho*(r - d[2]['pendulum_position'])**2 + d[2]['u']**2)/n_it
        obj_rel[2].append((obj2 - obj)*100/obj2)
    else:
        obj_rel[2].append(0)

# save results
headers = ['n_it', 'obj_rel_npgm', 'obj_rel_1pgm']
with open('mecotron_step_nPGM.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerow(headers)
    for k in range(len(n_iter)):
        row = [n_iter[k], obj_rel[2][k], obj_rel[1][k]]
        w.writerow(row)

# plot results
plt.figure()
lgs = ['1 PGM at 50*n Hz, N = 20', '1 PGM at 50*n Hz, N = 20*n', 'n PGM at 50 Hz, N = 20']
legend = []
for k in range(3):
    if approaches[k]:
        l, = plt.plot(n_iter, obj_rel[k], 'x-', label=lgs[k])
        legend.append(l)
plt.xlabel('n')
plt.ylabel('\delta V_{\infty}')
plt.legend(handles=legend)

tmax = 1
plt.figure()
plt.subplot(3, 1, 1)
legend = []
l, = plt.plot(ref['t'], ref['pendulum_position'], 'k--', label='1 PGM at 50 Hz, N = 20')
legend.append(l)
for ns, d in data.items():
    if approaches[0]:
        l, = plt.plot(d[0]['t'], d[0]['pendulum_position'], label='1 PGM at %d Hz, N = 20' % (50*ns))
        legend.append(l)
    if approaches[1]:
        l, = plt.plot(d[1]['t'], d[1]['pendulum_position'], label='1 PGM at %d Hz, N = %d' % (50*ns, N*ns))
        legend.append(l)
    if approaches[2]:
        l, = plt.plot(d[2]['t'], d[2]['pendulum_position'], label='%d PGM at 50 Hz, N = 20' % (ns))
        legend.append(l)
plt.xlim([0, tmax])
plt.ylabel('x_p (m)')
plt.legend(handles=legend)
plt.subplot(3, 1, 2)
plt.plot(ref['t'], ref['velocity'], 'k--')
for ns, d in data.items():
    for k in range(3):
        if approaches[k]:
            plt.plot(d[k]['t'], d[k]['velocity'])
plt.xlim([0, tmax])
plt.ylabel('velocity (m)')
plt.subplot(3, 1, 3)
plt.plot(ref['t'], ref['u'], 'k--')
for ns, d in data.items():
    for k in range(3):
        if approaches[k]:
            plt.plot(d[k]['t'], d[k]['u'])
plt.xlim([0, tmax])
plt.ylabel('u (m/s)')
plt.show()
