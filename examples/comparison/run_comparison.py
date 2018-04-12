#!/usr/bin/python

import matplotlib.pyplot as plt
import subprocess
import csv
import numpy as np


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


approaches = ['rtpgm', 'hpmpc', 'qpoases']
# approaches = ['rtpgm', 'pgm', 'hpmpc', 'qpoases']
# approaches = ['rtpgm', 'hpmpc']
base = 'mecotron_step'

print 'Simulating mpc:'
for ap in approaches:
    print '\t* %s' % ap
    subprocess.check_output("./build/%s_%s -v 0 -t 100" % (base, ap), shell=True, stderr=subprocess.STDOUT)

data = {}
for ap in approaches:
    data[ap] = get_dict('%s.csv' % ap)

print 'Results:'
for ap in approaches:
    print '\t* %s' % ap
    print '\t\t- average solve time = %f ms' % (sum([ts*1e-3 for ts in data[ap]['ts']])/len(data[ap]['ts']))
    if ap != 'rtpgm' and 'rtpgm' in approaches:
        rho = 1e5
        r = 0.12
        obj_ap = sum(rho*(r - data[ap]['pendulum_position'])**2 + data[ap]['u']**2)
        obj_rtpgm = sum(rho*(r - data['rtpgm']['pendulum_position'])**2 + data['rtpgm']['u']**2)
        print '\t\t- relative objective deviation of rtpgm = %f %%' % ((obj_rtpgm - obj_ap)*100/obj_ap)
# plot results
tmax = 1
plt.figure()
plt.subplot(5, 1, 1)
for ap in approaches:
    plt.plot(data[ap]['t'], data[ap]['pendulum_position'])
plt.xlim([0, tmax])
plt.ylabel('x_p (m)')
plt.legend(approaches)
plt.subplot(5, 1, 2)
for ap in approaches:
    plt.plot(data[ap]['t'], data[ap]['position'])
plt.xlim([0, tmax])
plt.ylabel('x (m)')
plt.subplot(5, 1, 3)
for ap in approaches:
    plt.plot(data[ap]['t'], data[ap]['theta'])
plt.xlim([0, tmax])
plt.ylabel('theta (m)')
plt.subplot(5, 1, 4)
for ap in approaches:
    plt.plot(data[ap]['t'], data[ap]['u'])
plt.xlim([0, tmax])
plt.ylabel('u (m/s)')
plt.subplot(5, 1, 5)
for ap in approaches:
    plt.semilogy(data[ap]['t'], data[ap]['ts']/1e6)
plt.xlim([0, tmax])
plt.xlabel('t (s)')
plt.ylabel('t_s (s)')

plt.show()
