# This file is part of rtPGM.
#
# rtPGM -- real-time Proximal Gradient Method
# Copyright (C) 2018 Ruben Van Parys, KU Leuven.
# All rights reserved.
#
# OMG-tools is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA


from controller import rtPGM, PGM, LQR
import numpy as np


def sat_cg(self, q='_q', nt=3):
    tabs = ''.join(['\t' for _ in range(nt)])
    ret = '%sfor (int k=0; k<%d; k++) {\n' % (tabs, self.N)
    ret += '%s\tif (%s[k] < %.8g) {\n' % (tabs, q, self.umin)
    ret += '%s\t\t%s[k] = %.8g;\n' % (tabs, q, self.umin)
    ret += '%s\t} else if (%s[k] > %.8g) {\n' % (tabs, q, self.umax)
    ret += '%s\t\t%s[k] = %.8g;\n' % (tabs, q, self.umax)
    ret += '%s\t}\n' % (tabs)
    ret += '%s}\n' % (tabs)
    return ret


def copy_cg(self, q_from, q_to, nt=3):
    tabs = ''.join(['\t' for _ in range(nt)])
    ret = '%sfor (int k=0; k<%d; k++) {\n' % (tabs, self.N)
    ret += '%s\t%s[k] = %s[k];\n' % (tabs, q_to, q_from)
    ret += '%s}\n' % (tabs)
    return ret


def copy_sat_cg(self, q_from, q_to, nt=3):
    tabs = ''.join(['\t' for _ in range(nt)])
    ret = '%sfor (int k=0; k<%d; k++) {\n' % (tabs, self.N)
    ret += '%s\t%s[k] = %s[k];\n' % (tabs, q_to, q_from)
    ret += '%s\tif (%s[k] < %.8g) {\n' % (tabs, q_to, self.umin)
    ret += '%s\t\t%s[k] = %.8g;\n' % (tabs, q_to, self.umin)
    ret += '%s\t} else if (%s[k] > %.8g) {\n' % (tabs, q_to, self.umax)
    ret += '%s\t\t%s[k] = %.8g;\n' % (tabs, q_to, self.umax)
    ret += '%s\t}\n' % (tabs)
    ret += '%s}\n' % (tabs)
    return ret


def mul_cg(A, b):
    ret = []
    for k in range(A.shape[0]):
        ret.append('')
        for l in range(A.shape[1]):
            ret[k] += '%.8g*%s[%d]' % (A[k, l], b, l)
            if l != A.shape[1]-1:
                ret[k] += ' + '
    return ret


def mul_cg2(A, b1, b2):
    ret = []
    for k in range(A.shape[0]):
        ret.append('')
        for l in range(A.shape[1]):
            ret[k] += '%.8g*(%s[%d] + %s[%d])' % (A[k, l], b1, l, b2, l)
            if l != A.shape[1]-1:
                ret[k] += ' + '
    return ret


def project_codegen(self, f):
    a = self.a
    aainv = self.aainv
    bx = self.Su.T.dot(self.aN)
    br = -self.Su.T.dot(self.aN.dot(self.Tx))

    n = a.shape[0]
    f.write('\t\tbool project(float* x, float* r) {\n')
    if n == 0 or not self.terminal_constraint:
        f.write(sat_cg(self))
        f.write('\t\t\t_n_it_proj = 1;\n')
        f.write('\t\t\treturn true;\n')
    elif n == 1:
        if self.integral_fb:
            nx = self.nx - self.ny
            f.write('\t\t\tfloat b = %s + %s + %s;\n' % (mul_cg(bx[:, :nx], 'x')[0], mul_cg(bx[:, nx:], '_e_int')[0], mul_cg(br, 'r')[0]))
        else:
            f.write('\t\t\tfloat b = %s + %s;\n' % (mul_cg(bx, 'x')[0], mul_cg(br, 'r')[0]))
        f.write('\t\t\tfloat lam0 = 0.0f;\n')
        f.write('\t\t\tfloat q[%d];\n' % self.N)
        f.write(copy_cg(self, '_q', 'q'))
        f.write(sat_cg(self, 'q'))
        f.write('\t\t\tfloat f0 = b + %s;\n' % (mul_cg(a, 'q')[0]))
        f.write('\t\t\tif (fabs(f0) < %.8g) {\n' % self.terminal_constraint_tol)
        f.write(copy_cg(self, 'q', '_q', 4))
        f.write('\t\t\t\treturn true;\n')
        f.write('\t\t\t}\n')
        f.write('\t\t\tfloat lam1 = %s + %.8g*b;\n' % (mul_cg(a*aainv, '_q')[0], aainv[0, 0]))
        f.write('\t\t\t')
        for k in range(self.N):
            f.write('q[%d] = _q[%d] - lam1*%.8g; ' % (k, k, a[0, k]))
        f.write('\n')
        f.write(sat_cg(self, 'q'))
        f.write('\t\t\tfloat f1 = b + %s;\n' % (mul_cg(a, 'q')[0]))
        f.write('\t\t\tint cnt = 0;\n')
        f.write('\t\t\tfloat lam2;\n')
        f.write('\t\t\twhile (fabs(f1) > %.8g) {\n' % self.terminal_constraint_tol)
        f.write('\t\t\t\tif (fabs(f1 - f0) < 1e-12) {\n')
        f.write('\t\t\t\t\treturn false;\n')
        f.write('\t\t\t\t}\n')
        f.write('\t\t\t\tlam2 = (lam0*f1 - lam1*f0)/(f1 - f0);\n')
        f.write('\t\t\t\tf0 = f1;\n')
        f.write('\t\t\t\tlam0 = lam1;\n')
        f.write('\t\t\t\tlam1 = lam2;\n')
        f.write('\t\t\t\t')
        for k in range(self.N):
            f.write('q[%d] = _q[%d] - lam1*%.8g; ' % (k, k, a[0, k]))
        f.write('\n')
        f.write(sat_cg(self, 'q', 4))
        f.write('\t\t\t\tf1 = b + %s;\n' % (mul_cg(a, 'q')[0]))
        f.write('\t\t\t\tcnt++;\n')
        f.write('\t\t\t}\n')
        f.write(copy_cg(self, 'q', '_q'))
        f.write('\t\t\t_n_it_proj = cnt;\n')
        f.write('\t\t\treturn true;\n')
    else:
        f.write('\t\t\tfloat b[%d];\n' % n)
        if self.integral_fb:
            nx = self.nx - self.ny
            b1a = mul_cg(bx[:, :nx], 'x')
            b1b = mul_cg(bx[:, nx:], '_e_int')
            b1 = [ba + ' + ' + bb for ba, bb in zip(b1a, b1b)]
        else:
            b1 = mul_cg(bx, 'x')
        b2 = mul_cg(br, 'r')
        for k in range(n):
            f.write('\t\t\tb[%d] = %s + %s;\n' % (k, b1[k], b2[k]))
        f.write('\t\t\tfloat lam0[%d];\n' % n)
        for k in range(n):
            f.write('\t\t\tlam0[%d] = 0.0f;\n' % (k))
        f.write('\t\t\tfloat q[%d];\n' % self.N)
        f.write(copy_sat_cg(self, '_q', 'q'))
        f.write('\t\t\tfloat f0[%d];\n' % n)
        f0 = mul_cg(a, 'q')
        for k in range(n):
            f.write('\t\t\tf0[%d] = b[%d] + %s;\n' % (k, k, f0[k]))
        f.write('\t\t\tif (fnorm(f0) < %.8g) {\n' % self.terminal_constraint_tol**2)
        f.write(copy_cg(self, 'q', '_q', 4))
        f.write('\t\t\t\treturn true;\n')
        f.write('\t\t\t}\n')
        f.write('\t\t\tfloat lam1[%d];\n' % n)
        lam1a = mul_cg(aainv.dot(a), '_q')
        lam1b = mul_cg(aainv, 'b')
        for k in range(n):
            f.write('\t\t\tlam1[%d] = %s + %s;\n' % (k, lam1a[k], lam1b[k]))
        t = mul_cg(a.T, 'lam1')
        f.write('\t\t\t')
        for k in range(self.N):
            f.write('q[%d] = _q[%d] - (%s); ' % (k, k, t[k]))
        f.write('\n')
        f.write(sat_cg(self, 'q'))
        f.write('\t\t\tfloat f1[%d];\n' % n)
        f1 = mul_cg(a, 'q')
        for k in range(n):
            f.write('\t\t\tf1[%d] = b[%d] + %s;\n' % (k, k, f1[k]))
        f.write('\t\t\tfloat a[%d];\n' % n)
        f.write('\t\t\tfloat df[%d];\n' % n)
        f.write('\t\t\tfloat dlam[%d];\n' % n)
        f.write('\t\t\tfloat lam2[%d];\n' % n)
        f.write('\t\t\tfloat dfdf;\n')
        f.write('\t\t\tfloat Jinv[%d][%d];\n' % (n, n))
        f.write('\t\t\t')
        for k in range(n):
            for l in range(n):
                if k == l:
                    f.write('Jinv[%d][%d] = 1.0f; ' % (k, l))
                else:
                    f.write('Jinv[%d][%d] = 0.0f; ' % (k, l))
        f.write('\n')
        f.write('\t\t\tint cnt = 0;\n')
        f.write('\t\t\twhile (fnorm(f1) > %.8g) {\n' % self.terminal_constraint_tol**2)
        for k in range(n):
            f.write('\t\t\t\tdlam[%d] = lam1[%d] - lam0[%d];\n' % (k, k, k))
        for k in range(n):
            f.write('\t\t\t\tdf[%d] = f1[%d] - f0[%d];\n' % (k, k, k))
        f.write('\t\t\t\tdfdf = ')
        for k in range(n):
            f.write('df[%d]*df[%d]' % (k, k))
            if k != n-1:
                f.write(' + ')
            else:
                f.write(';\n')
        f.write('\t\t\t\tif (dfdf < 1e-14) {\n')
        f.write('\t\t\t\t\treturn false;\n')
        f.write('\t\t\t\t}\n')
        for k in range(n):
            f.write('\t\t\t\ta[%d] = (dlam[%d] ' % (k, k))
            for l in range(n):
                f.write(' - Jinv[%d][%d]*df[%d]' % (k, l, l))
            f.write(')/dfdf;\n')
        f.write('\t\t\t\t')
        for k in range(n):
            for l in range(n):
                f.write('Jinv[%d][%d] += a[%d]*df[%d]; ' % (k, l, k, l))
        f.write('\n')
        for k in range(n):
            f.write('\t\t\t\tlam2[%d] = lam1[%d]' % (k, k))
            for l in range(n):
                f.write(' - Jinv[%d][%d]*f1[%d]' % (k, l, l))
            f.write(';\n')
        for k in range(n):
            f.write('\t\t\t\tf0[%d] = f1[%d];\n' % (k, k))
        for k in range(n):
            f.write('\t\t\t\tlam0[%d] = lam1[%d];\n' % (k, k))
        for k in range(n):
            f.write('\t\t\t\tlam1[%d] = lam2[%d];\n' % (k, k))

        t = mul_cg(a.T, 'lam1')
        f.write('\t\t\t\t')
        for k in range(self.N):
            f.write('q[%d] = _q[%d] - (%s); ' % (k, k, t[k]))
        f.write('\n')
        f.write(sat_cg(self, 'q', 4))
        f1 = mul_cg(a, 'q')
        for k in range(n):
            f.write('\t\t\t\tf1[%d] = b[%d] + %s;\n' % (k, k, f1[k]))
        f.write('\t\t\t\tcnt++;\n')
        f.write('\t\t\t}\n')
        f.write(copy_cg(self, 'q', '_q'))
        f.write('\t\t\t_n_it_proj = cnt;\n')
        f.write('\t\t\treturn true;\n')
    f.write('\t\t}\n')


def gradientstep_codegen(self, f):
    f.write('\t\tvoid gradient_step(float* x, float* r) {\n')
    t1 = mul_cg(np.eye(self.N*self.nu) - self.gamma*self.F, 'q_prev')
    if self.integral_fb:
        nx = self.nx - self.ny
        t2a = mul_cg(-self.gamma*self.H[:, :nx], 'x')
        t2b = mul_cg(-self.gamma*self.H[:, nx:], '_e_int')
        t2 = [ta + ' + ' + tb for ta, tb in zip(t2a, t2b)]
    else:
        t2 = mul_cg(-self.gamma*self.H, 'x')
    t3 = mul_cg(self.gamma*self.H.dot(self.Tx), 'r')
    f.write('\t\t\tfloat q_prev[%d];\n' % (self.N))
    f.write(copy_cg(self, '_q', 'q_prev'))
    for k in range(self.N):
        f.write('\t\t\t_q[%d] = %s + %s + %s;\n' % (k, t1[k], t2[k], t3[k]))
    f.write('\t\t}\n')


def update_rtPGM_codegen(self, f):
    f.write('\t\tbool update(float* x, float* r, float* u) {\n')
    f.write('\t\t\tgradient_step(x, r);\n')
    f.write('\t\t\tif (!project(x, r)) {\n')
    f.write('\t\t\t\treturn false;\n')
    f.write('\t\t\t}\n')
    f.write('\t\t\t*u = _q[0];\n')
    f.write('\t\t\tshift();\n')
    if self.integral_fb:
        y = mul_cg(self.C[:, :self.nx - self.ny], 'x')
        for k in range(self.ny):
            f.write('\t\t\t_e_int[%d] += %s - r[%d];\n' % (k, y[k], k))
    f.write('\t\t\treturn true;\n')
    f.write('\t\t}\n')


def shift_codegen(self, f):
    f.write('\t\tvoid shift() {\n')
    f.write('\t\t\tfor (int k=0; k<%d; k++) {\n' % (self.N-1))
    f.write('\t\t\t\t_q[k] = _q[k+1];\n')
    f.write('\t\t\t}\n')
    f.write('\t\t\t_q[%d] = 0.0f;\n' % (self.N-1))
    f.write('\t\t}\n')


def rtPGM_codegen(self, path='rtPGM.h'):
    f = open(path, 'w')
    include_guard = path.split('.')[0].split('/')[-1].upper() + '_H'
    f.write('#ifndef %s\n' % include_guard)
    f.write('#define %s\n' % include_guard)
    f.write('class rtPGM {\n')
    f.write('\tprivate:\n')
    f.write('\t\tfloat _q[%d];\n' % self.N*self.nu)
    f.write('\t\tint _n_it_proj;\n')
    if self.integral_fb:
        f.write('\t\tfloat _e_int[%d];\n' % self.ny)
    f.write('\n')
    f.write('\t\tfloat fabs(float value) {\n')
    f.write('\t\t\treturn (value >= 0.0f) ? value : -value;\n')
    f.write('\t\t}\n\n')
    f.write('\t\tfloat fnorm(float* value) {\n')
    n = self.a.shape[0]
    f.write('\t\t\treturn ')
    for k in range(n):
        f.write('value[%d]*value[%d]' % (k, k))
        if k != n-1:
            f.write(' + ')
        else:
            f.write(';\n')
    f.write('\t\t}\n\n')
    gradientstep_codegen(self, f)
    f.write('\n')
    project_codegen(self, f)
    f.write('\n')
    shift_codegen(self, f)
    f.write('\n')
    f.write('\tpublic:\n')
    f.write('\t\trtPGM() : _n_it_proj(0) { reset(); }\n\n')
    f.write('\t\tvoid reset() {\n')
    f.write('\t\t\tfor (int k=0; k<%d; k++) {\n' % self.N)
    f.write('\t\t\t\t_q[k] = 0.0f;\n')
    f.write('\t\t\t}\n')
    if self.integral_fb:
        for k in range(self.ny):
            f.write('\t\t\t_e_int[%d] = 0.0f;\n' % (k))
    f.write('\t\t}\n\n')
    f.write('\t\tint N() {\n')
    f.write('\t\t\treturn %d;\n' % self.N)
    f.write('\t\t}\n\n')
    f.write('\t\tint n_it_proj() {\n')
    f.write('\t\t\treturn _n_it_proj;\n')
    f.write('\t\t}\n\n')
    f.write('\t\tvoid input_trajectory(float* q) {\n')
    f.write(copy_cg(self, '_q', 'q'))
    f.write('\t\t}\n\n')
    update_rtPGM_codegen(self, f)
    f.write('\n')
    f.write('};\n')
    f.write('#endif\n')
    f.close()


def PGM_codegen(self, path='PGM.h'):
    f = open(path, 'w')
    include_guard = path.split('.')[0].upper() + '_H'
    include_guard = path.split('.')[0].split('/')[-1].upper() + '_H'
    f.write('#ifndef %s\n' % include_guard)
    f.write('#define %s\n' % include_guard)
    f.write('class PGM {\n')
    f.write('\tprivate:\n')
    f.write('\t\tfloat _q[%d];\n' % self.N*self.nu)
    f.write('\t\tint _n_it_proj;\n')
    if self.integral_fb:
        f.write('\t\tfloat _e_int[%d];\n' % self.ny)
    f.write('\n')
    f.write('\t\tfloat fabs(float value) {\n')
    f.write('\t\t\treturn (value >= 0.0f) ? value : -value;\n')
    f.write('\t\t}\n\n')
    f.write('\t\tfloat fnorm(float* value) {\n')
    n = self.a.shape[0]
    f.write('\t\t\treturn ')
    for k in range(n):
        f.write('value[%d]*value[%d]' % (k, k))
        if k != n-1:
            f.write(' + ')
        else:
            f.write(';\n')
    f.write('\t\t}\n\n')
    gradientstep_codegen(self, f)
    f.write('\n')
    project_codegen(self, f)
    f.write('\n')
    shift_codegen(self, f)
    f.write('\n')
    residual_codegen(self, f)
    f.write('\n')
    f.write('\tpublic:\n')
    f.write('\t\tPGM() : _n_it_proj(0) { reset(); }\n\n')
    f.write('\t\tvoid reset() {\n')
    f.write('\t\t\tfor (int k=0; k<%d; k++) {\n' % self.N)
    f.write('\t\t\t\t_q[k] = 0.0f;\n')
    f.write('\t\t\t}\n')
    if self.integral_fb:
        for k in range(self.ny):
            f.write('\t\t\t_e_int[%d] = 0.0f;\n' % (k))
    f.write('\t\t}\n\n')
    f.write('\t\tint N() {\n')
    f.write('\t\t\treturn %d;\n' % self.N)
    f.write('\t\t}\n\n')
    f.write('\t\tint n_it_proj() {\n')
    f.write('\t\t\treturn _n_it_proj;\n')
    f.write('\t\t}\n\n')
    f.write('\t\tvoid input_trajectory(float* q) {\n')
    f.write(copy_cg(self, '_q', 'q'))
    f.write('\t\t}\n\n')
    update_PGM_codegen(self, f)
    f.write('\n')
    f.write('};\n')
    f.write('#endif\n')
    f.close()


def residual_codegen(self, f):
    f.write('\t\tfloat residual(float* q1, float* q0) {\n')
    y = mul_cg2((self.F - (1./self.gamma)*np.eye(self.N*self.nu)), 'q1', '-q0')
    f.write('\t\t\tfloat res = 0;\n')
    f.write('\t\t\tfloat dres;\n')
    for k in range(self.nu*self.N):
        f.write('\t\t\tdres = %s;\n' % y[k])
        f.write('\t\t\tres += dres*dres;\n')
    f.write('\t\t\treturn res;\n')
    f.write('\t\t}\n')


def update_PGM_codegen(self, f):
    f.write('\t\tbool update(float* x, float* r, float* u) {\n')
    f.write('\t\t\tint cnt = 0;\n')
    f.write('\t\t\tfloat q0[%d];\n' % self.N)
    f.write('\t\t\twhile (true) {\n')
    f.write(copy_cg(self, '_q', 'q0', nt=4))
    f.write('\t\t\t\tgradient_step(x, r);\n')
    f.write('\t\t\t\tif (!project(x, r)) {\n')
    f.write('\t\t\t\t\treturn false;\n')
    f.write('\t\t\t\t}\n')
    f.write('\t\t\t\tif (++cnt > %d) {\n' % self.max_iter)
    f.write('\t\t\t\t\tbreak;\n')
    f.write('\t\t\t\t}\n')
    f.write('\t\t\t\tif (residual(_q, q0) < %.8g) {\n' % self.tol**2)
    f.write('\t\t\t\t\tbreak;\n')
    f.write('\t\t\t\t}\n')
    f.write('\t\t\t}\n')
    f.write('\t\t\t*u = _q[0];\n')
    f.write('\t\t\tshift();\n')
    if self.integral_fb:
        y = mul_cg(self.C[:, :self.nx - self.ny], 'x')
        for k in range(self.ny):
            f.write('\t\t\t_e_int[%d] += %s - r[%d];\n' % (k, y[k], k))
    f.write('\t\t\treturn true;\n')
    f.write('\t\t}\n')


def update_LQR_codegen(self, f):
    f.write('\t\tbool update(float* x, float* r, float* u) {\n')
    if not self.integral_fb:
        u1 = mul_cg(-self.Klqr, 'x')
        u2 = mul_cg(self.Klqr.dot(self.Tx)+self.Tu, 'r')
        if self.Klqr.shape[0] == 1:
            f.write('\t\t\t*u = %s + %s;\n' % (u1[0], u2[0]))
        else:
            for k in range(self.Klqr.shape[0]):
                f.write('\t\t\tu[%d] = %s + %s;\n' % (u1[k], u2[k]))
    else:
        nx = self.nx - self.ny
        u1a = mul_cg(-self.Klqr[:, :nx], 'x')
        u1b = mul_cg(-self.Klqr[:, nx:], '_e_int')
        u2 = mul_cg(self.Klqr[:, :nx].dot(self.Tx[:nx, :]+self.Tu[:nx, :]), 'r')
        if self.Klqr.shape[0] == 1:
            f.write('\t\t\t*u = %s + %s + %s;\n' % (u1a[0], u1b[0], u2[0]))
        else:
            for k in range(self.Klqr.shape[0]):
                f.write('\t\t\tu[%d] = %s + %s + %s;\n' % (u1a[k], u1b[k], u2[k]))
        y = mul_cg(self.C[:, :nx], 'x')
        for k in range(self.ny):
            f.write('\t\t\t_e_int[%d] += %s - r[%d];\n' % (k, y[k], k))
    f.write('\t\t\treturn true;\n')
    f.write('\t\t}\n\n')


def LQR_codegen(self, path='LQR.h'):
    f = open(path, 'w')
    include_guard = path.split('.')[0].split('/')[-1].upper() + '_H'
    f.write('#ifndef %s\n' % include_guard)
    f.write('#define %s\n' % include_guard)
    f.write('class LQR {\n')
    f.write('\tprivate:\n')
    if self.integral_fb:
        f.write('\t\tfloat _e_int[%d];\n' % self.ny)
    f.write('\tpublic:\n')
    f.write('\t\tLQR() { reset(); }\n\n')
    f.write('\t\tvoid reset() {\n')
    if self.integral_fb:
        f.write('\t\t\tfor (int k=0; k<%d; k++) {\n' % self.ny)
        f.write('\t\t\t\t_e_int[k] = 0.0f;\n')
        f.write('\t\t\t}\n')
    f.write('\t\t}\n\n')
    update_LQR_codegen(self, f)
    f.write('};\n')
    f.write('#endif\n')
    f.close()

rtPGM.codegen = rtPGM_codegen
PGM.codegen = PGM_codegen
LQR.codegen = LQR_codegen
