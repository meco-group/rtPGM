# This file is part of rtPGM.
#
# rtPGM -- real-time Proximal Gradient Method
# Copyright (C) 2018 Ruben Van Parys, KU Leuven.
# All rights reserved.
#
# rtPGM is free software; you can redistribute it and/or
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


import numpy as np
import scipy.linalg
import scipy.signal
import numpy.linalg as la


def iuc(eigv):
    if np.linalg.norm(eigv) < 1:
        return True
    else:
        return False


def split_modes(A):
    J, S, sdim = scipy.linalg.schur(A, output='real', sort=iuc)
    ns = sdim
    return J, S, ns


class StateFeedback:
    def __init__(self, A, B, C, D, Q, R, S=None, **kwargs):
        StateFeedback.process_kwargs(self, **kwargs)
        self.check_input_data(A, B, C, D, Q, R, S)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        self.ny = self.C.shape[0]
        if self.nu != 1:
            raise ValueError('Currently only single-input systems are supported.')
        # relation between reference and steady-state state & input
        Tinv = np.vstack((np.hstack((self.A - np.eye(self.nx), self.B)),
                          np.hstack((self.C, np.zeros((self.C.shape[0], self.nu))))))
        T = np.linalg.solve(Tinv, np.eye(Tinv.shape[0])).dot(
            np.vstack((np.zeros((self.nx, 1)), np.ones((self.ny, 1)))))
        self.Tx = T[:self.nx, :]
        self.Tu = T[self.nx:, :]
        if self.integral_fb:
            self.A = np.vstack((np.hstack((A, np.zeros((self.nx, self.ny)))),
                                np.hstack((C, np.eye(self.ny)))))
            self.B = np.vstack((B, self.D))
            self.C = np.hstack((C, np.zeros((self.ny, self.ny))))
            self.nx += self.ny
            self.e_int = np.zeros((self.ny, 1))
            self.Q = scipy.linalg.block_diag(
                self.Q, self.integral_weight*np.eye(self.ny))
            self.Tx = np.vstack((self.Tx, np.zeros((self.ny, self.ny))))
        self.S = np.zeros((self.nu, self.nx)) if S is None else S

    def process_kwargs(self, **kwargs):
        if 'integral_fb' in kwargs:
            self.integral_fb = kwargs['integral_fb']
        else:
            self.integral_fb = False
        if 'integral_weight' in kwargs:
            self.integral_weight = kwargs['integral_weight']
        else:
            self.integral_weight = False

    def check_input_data(self, A, B, C, D, Q, R, S):
        if (A.shape[0] != A.shape[1]):
            raise ValueError('A matrix has invalid size!')
        if (B.shape[0] != A.shape[1]):
            raise ValueError('A and B matrices have incompatible sizes!')
        if (C.shape[1] != A.shape[1]):
            raise ValueError('A and C matrices have incompatible sizes!')
        if (D.shape[0] != C.shape[0]):
            raise ValueError('C and D matrices have incompatible sizes!')
        if (D.shape[1] != B.shape[1]):
            raise ValueError('B and D matrices have incompatible sizes!')
        if (not np.allclose(Q, Q.T, atol=1e-8)):
            raise ValueError('Q is not symmetric!')
        if (not np.allclose(R, R.T, atol=1e-8)):
            raise ValueError('Q is not symmetric!')
        if (not np.all(np.round(np.linalg.eigvals(Q), 8) >= 0)):
            raise ValueError('Q is not positive semi-definite!')
        if (not np.all(np.round(np.linalg.eigvals(R), 8) > 0)):
            raise ValueError('R is not positive definite!')
        if (Q.shape[0] != A.shape[0]):
            raise ValueError('Q matrix has invalid size!')
        if (R.shape[0] != B.shape[1]):
            raise ValueError('R matrix has invalid size!')
        if S is None:
            return
        if (S.shape[0] != D.shape[1] or S.shape[1] != C.shape[1]):
            raise ValueError('D matrix has invalid size!')

    def update(self, x, r=None):
        raise NotImplementedError('Please implement this method!')

    def reset(self):
        if self.integral_fb:
            self.e_int = np.zeros((self.ny, 1))


class LQR(StateFeedback):
    def __init__(self, A, B, C, D, Q, R, S=None, **kwargs):
        StateFeedback.__init__(self, A, B, C, D, Q, R, S, **kwargs)
        if S is not None:
            print 'Neglecting S for LQR...'
        X = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.Klqr = scipy.linalg.inv(self.B.T.dot(X).dot(self.B) +
                                     self.R)*(self.B.T.dot(X).dot(self.A))

    def update(self, x, r=None):
        if r is None:
            r = np.zeros((self.Tx.shape[1], 1))
        if self.integral_fb:
            x = np.vstack((x, self.e_int))
            self.e_int += self.C[:, :x.shape[0]].dot(x) - r
        return -self.Klqr.dot(x - self.Tx.dot(r))

    def VN(self, x, r=None, q=None):
        return 0

    def place_poles(self, poles):
        fsf = scipy.signal.place_poles(self.A, self.B, poles)
        self.Klqr = fsf.gain_matrix


class SaturatedLQR(LQR):
    def __init__(self, A, B, C, D, Q, R, umin, umax, S=None, **kwargs):
        LQR.__init__(self, A, B, C, D, Q, R, S, **kwargs)
        self.umin = umin
        self.umax = umax

    def update(self, x, r=None):
        u = LQR.update(self, x, r)
        if u < self.umin:
            return np.c_[self.umin]
        if u > self.umax:
            return np.c_[self.umax]
        return u


class MPC(StateFeedback):
    def __init__(self, A, B, C, D, Q, R, N=100, S=None, **kwargs):
        StateFeedback.__init__(self, A, B, C, D, Q, R, S, **kwargs)
        self.N = N
        MPC.process_kwargs(self, **kwargs)
        self.build()
        self.reset()

    def process_kwargs(self, **kwargs):
        if 'Pf' in kwargs:
            Pf = kwargs['Pf']
            if (not np.allclose(Pf, Pf.T, atol=1e-8)):
                raise ValueError('Pf is not symmetric!')
            if (not np.all(np.linalg.eigvals(Pf) > 0)):
                raise ValueError('Pf is not positive definite!')
            if (Pf.shape[0] != self.nx):
                raise ValueError('Pf matrix has invalid size!')
            self.Pf = kwargs['Pf']
        else:
            self.Pf = None
        if 'Kf' in kwargs:
            self.Kf = kwargs['Kf']
        else:
            self.Kf = None
        if 'Q_extra' in kwargs:
            Q_extra = kwargs['Q_extra']
            if (not np.allclose(Q_extra, Q_extra.T, atol=1e-8)):
                raise ValueError('Q_extra is not symmetric!')
            if (not np.all(np.round(np.linalg.eigvals(Q_extra), 5) >= 0)):
                raise ValueError('Q_extra is not positive semi-definite!')
            if (Q_extra.shape[0] != self.nx*self.N):
                raise ValueError('Q_extra has invalid size!')
            self.Q_extra = kwargs['Q_extra']
        else:
            self.Q_extra = None
        if 'R_extra' in kwargs:
            R_extra = kwargs['R_extra']
            if (not np.allclose(R_extra, R_extra.T, atol=1e-8)):
                raise ValueError('R_extra is not symmetric!')
            if (not np.all(np.round(np.linalg.eigvals(R_extra), 5) >= 0)):
                raise ValueError('R_extra is not positive semi-definite!')
            if (R_extra.shape[0] != self.nu*self.N):
                raise ValueError('R_extra has invalid size!')
            self.R_extra = kwargs['R_extra']
        else:
            self.R_extra = None
        if 'S_extra' in kwargs:
            S_extra = kwargs['S_extra']
            if (S_extra.shape[0] != self.nu*self.N or S_extra.shape[1] != self.nx*self.N):
                raise ValueError('S_extra has invalid size!')
            self.S_extra = kwargs['S_extra']
        else:
            self.S_extra = None

    def reset(self):
        StateFeedback.reset(self)
        self.q = np.zeros((self.N, 1))

    def build(self):
        # split stable and unstable modes
        J, S, self.ns = scipy.linalg.schur(self.A, output='real', sort=iuc)
        self.Ss, self.Su = S[:, :self.ns], S[:, self.ns:]
        self.Js, self.Ju = J[:self.ns, :self.ns], J[self.ns:, self.ns:]
        # compute terminal cost & terminal state feedback controller
        if self.Pf is None and self.Kf is None:
            self.Kf = np.zeros((self.nu, self.nx))
            if self.Js.shape[0] > 0:
                Ps = scipy.linalg.solve_discrete_lyapunov(
                    self.Js.T, self.Ss.T.dot(self.Q).dot(self.Ss))
            else:
                Ps = np.zeros((0, 0))
            self.Pf = S.dot(scipy.linalg.block_diag(Ps, np.zeros(
                (self.nx-self.ns, self.nx-self.ns)))).dot(S.T)
        elif self.Pf is None:
            Ak = self.A - self.B.dot(self.Kf)
            Qk = self.Q + self.Kf.T.dot(self.R).dot(self.Kf)
            self.Pf = scipy.linalg.solve_discrete_lyapunov(Ak.T, Qk)
        else:
            self.Pf = self.Pf
            self.Kf = self.Kf
        # generate QP matrices
        _Q = scipy.linalg.block_diag(*[self.Q for _ in range(self.N)])
        _R = scipy.linalg.block_diag(*[self.R for _ in range(self.N)])
        _S = scipy.linalg.block_diag(*[self.S for _ in range(self.N)])
        if self.Q_extra is not None:
            _Q += self.Q_extra
        if self.R_extra is not None:
            _R += self.R_extra
        if self.S_extra is not None:
            _S += self.S_extra
        a = np.vstack([la.matrix_power(self.A, k) for k in range(self.N+1)])
        b = np.zeros((self.nx, self.N*self.nu))
        for k in range(self.N):
            row = np.zeros((self.nx, self.N*self.nu))
            for l in range(k+1):
                row[:, l] = la.matrix_power(self.A, k-l).dot(self.B).ravel()
            b = np.vstack((b, row))
        self.b0 = b[:self.N*self.nx]
        self.a0 = a[:self.N*self.nx]
        self.bN = b[self.N*self.nx:(self.N+1)*self.nx]
        self.aN = a[self.N*self.nx:(self.N+1)*self.nx]
        self.F = _R + self.b0.T.dot(_Q).dot(self.b0) + self.bN.T.dot(
            self.Pf).dot(self.bN) + _S.dot(self.b0) + self.b0.T.dot(_S.T)
        self.G = self.a0.T.dot(_Q).dot(self.a0) + \
            self.aN.T.dot(self.Pf).dot(self.aN)
        self.H = self.b0.T.dot(_Q).dot(
            self.a0) + self.bN.T.dot(self.Pf).dot(self.aN) + _S.dot(self.a0)

    def VN(self, x, r=None, q=None):
        q = self.q if q is None else q
        r = np.zeros((self.Tx.shape[1], 1)) if r is None else r
        if self.integral_fb:
            x = np.vstack((x, self.e_int))
        dx = x - self.Tx.dot(r)
        dq = q - self.Tu.dot(r)
        return float(0.5*dq.T.dot(self.F).dot(dq) +
                     0.5*dx.T.dot(self.G).dot(dx) + dq.T.dot(self.H).dot(dx))

    def update(self, x, r=None):
        raise NotImplementedError('Please implement this method!')


class rtPGM(MPC):
    def __init__(self, A, B, C, D, Q, R, umin, umax, N=100, S=None, **kwargs):
        MPC.__init__(self, A, B, C, D, Q, R, N, S, **kwargs)
        self.umin = umin
        self.umax = umax
        rtPGM.process_kwargs(self, **kwargs)
        # compute affine set {q|aq=b} representing terminal set constraint
        # (b depends on x & r)
        self.a = self.Su.T.dot(self.bN)
        self.aainv = np.linalg.solve(
            self.a.dot(self.a.T), np.eye(self.nx-self.ns))
        # compute gamma
        eigv = np.linalg.eigvals(self.F)
        l_max = max(eigv)
        self.gamma = self.gamma_ratio*2./l_max

    def process_kwargs(self, **kwargs):
        if 'gamma_ratio' in kwargs:
            self.gamma_ratio = kwargs['gamma_ratio']
        else:
            self.gamma_ratio = 0.99
        if 'terminal_constraint' in kwargs:
            self.terminal_constraint = kwargs['terminal_constraint']
        else:
            self.terminal_constraint = True
        if 'terminal_constraint_tol' in kwargs:
            self.terminal_constraint_tol = kwargs['terminal_constraint_tol']
        else:
            self.terminal_constraint_tol = 1e-3

    def update(self, x, r=None):
        r = np.zeros((self.Tx.shape[1], 1)) if r is None else r
        if self.integral_fb:
            x = np.vstack((x, self.e_int))
        q_grad = (np.eye(self.N*self.nu) - self.gamma*self.F).dot(self.q) - \
            self.gamma*(self.H.dot(x - self.Tx.dot(r)))
        self.q = self.project(q_grad, x, r, self.umin,
                              self.umax, self.terminal_constraint_tol)
        u = np.c_[self.q[0, 0]]
        # compute xN
        self.xN = np.linalg.norm(self.Su.T.dot(
            self.aN.dot(x - self.Tx.dot(r)) + self.bN.dot(self.q)))
        if self.integral_fb:
            self.e_int += self.C[:, :x.shape[0]].dot(x) - r
        # shift
        xN = self.aN.dot(x - self.Tx.dot(r)) + self.bN.dot(self.q)
        self.q = np.vstack((self.q[1:, :], -self.Kf.dot(xN)))
        return u

    def sat(self, q, umin, umax):
        ret = np.copy(q)
        for k in range(q.shape[0]):
            if q[k, 0] < umin:
                ret[k, 0] = umin
            if q[k, 0] > umax:
                ret[k, 0] = umax
        return ret

    def project(self, q, x, r, umin, umax, tol=1e-3, verbose=0, good_broyden=False):
        a = self.a
        aainv = self.aainv
        b = self.Su.T.dot(- self.aN.dot(x - self.Tx.dot(r)))
        n = a.shape[0]
        if n == 0 or not self.terminal_constraint:
            return self.sat(q, umin, umax)
        elif n == 1:  # secant method
            lam0 = 0.  # current point
            q0 = self.sat(q, umin, umax)
            f0 = a.dot(q0) - b
            if abs(f0) < tol:
                return q0
            lam1 = aainv.dot(a.dot(q) - b)  # projection on hp
            q1 = self.sat(q - lam1*a.T, umin, umax)
            f1 = a.dot(q1) - b
            cnt = 0
            while abs(f1) > tol:
                if abs(f1 - f0) < 1e-12:
                    raise ValueError('Infeasible problem!')
                lam2 = (lam0*f1 - lam1*f0) / (f1 - f0)
                q0, f0, lam0 = q1, f1, lam1
                lam1 = lam2
                q1 = self.sat(q - lam1*a.T, umin, umax)
                f1 = a.dot(q1) - b
                cnt = cnt + 1
            if verbose > 0:
                print 'n_it (secant): %d' % cnt
            return q1
        else:  # broyder's method
            lam0 = np.zeros((n, 1))  # current point
            q0 = self.sat(q, umin, umax)
            f0 = a.dot(q0) - b
            if np.linalg.norm(f0, 2) < tol:
                return q0
            lam1 = aainv.dot(a.dot(q) - b)  # projection on affine set
            q1 = self.sat(q - a.T.dot(lam1), umin, umax)
            f1 = a.dot(q1) - b
            Jinv = np.eye(n)
            cnt = 0
            while np.linalg.norm(f1, 2) > tol:
                dlam = lam1 - lam0
                df = f1 - f0
                if np.linalg.norm(df, 2) < 1e-12:
                    raise ValueError('Infeasible problem!')
                if good_broyden:  # good broyden
                    Jinv = Jinv + \
                        ((dlam - Jinv.dot(df)) /
                         (dlam.T.dot(Jinv).dot(df))).dot(dlam.T.dot(Jinv))
                else:  # bad broyden
                    Jinv = Jinv + ((dlam - Jinv.dot(df)) /
                                   (df.T.dot(df))).dot(df.T)
                lam2 = lam1 - Jinv.dot(f1)
                q0, f0, lam0 = q1, f1, lam1
                lam1 = lam2
                q1 = self.sat(q - a.T.dot(lam1), umin, umax)
                f1 = a.dot(q1) - b
                cnt = cnt + 1
            if verbose > 0:
                print 'n_it (broyder): %d' % cnt
            return q1


class PGM(rtPGM):
    def __init__(self, A, B, C, D, Q, R, umin, umax, N=100, S=None, **kwargs):
        rtPGM.__init__(self, A, B, C, D, Q, R, umin, umax, N, S, **kwargs)
        PGM.process_kwargs(self, **kwargs)
        self.n_iter = []

    def process_kwargs(self, **kwargs):
        if 'tol' in kwargs:
            self.tol = kwargs['tol']
        else:
            self.tol = 1e-4
        if 'max_iter' in kwargs:
            self.max_iter = kwargs['max_iter']
        else:
            self.max_iter = 1000

    def reset(self):
        self.n_iter = []
        rtPGM.reset(self)

    def residual(self, dq):
        return (self.F - (1./self.gamma)*np.eye(self.N*self.nu)).dot(dq)

    def update(self, x, r=None, verbose=0):
        r = np.zeros((self.Tx.shape[1], 1)) if r is None else r
        if self.integral_fb:
            x = np.vstack((x, self.e_int))
        eps = np.c_[1000.]
        cnt = 0
        while (np.linalg.norm(eps, 2) > self.tol):
            q0 = np.copy(self.q)
            q_grad = (np.eye(self.N*self.nu) - self.gamma*self.F).dot(self.q) - \
                self.gamma*(self.H.dot(x - self.Tx.dot(r)))
            self.q = self.project(q_grad, x, r, self.umin,
                                  self.umax, self.terminal_constraint_tol)
            eps = self.residual(self.q - q0)
            cnt = cnt + 1
            if cnt >= self.max_iter:
                print 'Maximum iterations (%d) exceeded!' % self.max_iter
                break
        if verbose > 0:
            print 'n_iter = %d' % cnt
        self.n_iter.append(cnt)
        u = np.c_[self.q[0, 0]]
        # compute xN
        self.xN = np.linalg.norm(self.Su.T.dot(
            self.aN.dot(x - self.Tx.dot(r)) + self.bN.dot(self.q)))
        if self.integral_fb:
            self.e_int += self.C[:, :x.shape[0]].dot(x) - r
        # shift
        xN = self.aN.dot(x - self.Tx.dot(r)) + self.bN.dot(self.q)
        self.q = np.vstack((self.q[1:, :], -self.Kf.dot(xN)))
        return u
