import csv
from scipy.signal import cont2discrete
import csv
import numpy as np
from rtPGM.controller import rtPGM, PGM


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


def dump_model(model, A, B, Q, R, x_ss):
    mat = [A, B, Q, R, x_ss]
    with open(model, 'w') as f:
        for m in mat:
            f.write('%d %d\n' % (m.shape[0], m.shape[1]))
            for k in range(m.shape[0]):
                for l in range(m.shape[1]):
                    f.write('%.9f ' % m[k, l])
                f.write('\n')


def lti_codegen(A, B, C, D, Q, R, umin, umax, Tx, CN, Ts, name='lti', path=None):
    if path is None:
        path = name + '.h'
    nx = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]
    nxu = CN.shape[0]
    f = open(path, 'w')
    include_guard = name.upper() + '_H'
    f.write('#ifndef %s\n' % include_guard)
    f.write('#define %s\n' % include_guard)
    f.write('class %s {\n' % name.capitalize())
    f.write('\tprivate:\n')
    f.write('\t\tdouble _x[%d];\n' % nx)
    f.write('\t\tdouble _A[%d] = {' % (nx*nx))
    for k in range(nx):
        for l in range(nx):
            f.write('%f' % A[l, k])
            if (k == nx-1 and l == nx-1):
                f.write('};\n')
            else:
                f.write(',')
    f.write('\t\tdouble _B[%d] = {' % (nx*nu))
    for k in range(nu):
        for l in range(nx):
            f.write('%f' % B[l, k])
            if (k == nu-1 and l == nx-1):
                f.write('};\n')
            else:
                f.write(',')
    f.write('\t\tdouble _C[%d] = {' % (nx*ny))
    for k in range(nx):
        for l in range(ny):
            f.write('%f' % C[l, k])
            if (k == nx-1 and l == ny-1):
                f.write('};\n')
            else:
                f.write(',')
    f.write('\t\tdouble _D[%d] = {' % (nu*ny))
    for k in range(nu):
        for l in range(ny):
            f.write('%f' % D[l, k])
            if (k == nu-1 and l == ny-1):
                f.write('};\n')
            else:
                f.write(',')
    f.write('\t\tdouble _Q[%d] = {' % (nx*nx))
    for k in range(nx):
        for l in range(nx):
            f.write('%f' % Q[l, k])
            if (k == nx-1 and l == nx-1):
                f.write('};\n')
            else:
                f.write(',')
    f.write('\t\tdouble _R[%d] = {' % (nu*nu))
    for k in range(nu):
        for l in range(nu):
            f.write('%f' % R[l, k])
            if (k == nu-1 and l == nu-1):
                f.write('};\n')
            else:
                f.write(',')
    f.write('\t\tdouble _CN[%d] = {' % (nxu*nx))
    for k in range(nx):
        for l in range(nxu):
            f.write('%f' % CN[l, k])
            if (k == nx-1 and l == nxu-1):
                f.write('};\n')
            else:
                f.write(',')
    f.write('\t\tdouble _umin[%d] = {' % nu)
    for k in range(nu):
        f.write('%f' % umin[k])
        if (k == nu-1):
            f.write('};\n')
        else:
            f.write(',')
    f.write('\t\tdouble _umax[%d] = {' % nu)
    for k in range(nu):
        f.write('%f' % umax[k])
        if (k == nu-1):
            f.write('};\n')
        else:
            f.write(',')
    f.write('\t\tdouble _Ts = %f;\n' % Ts)

    f.write('\tpublic:\n')
    f.write('\t\t%s() {\n' % name.capitalize())
    f.write('\t\t\treset();\n')
    f.write('\t\t}\n')
    f.write('\t\tvoid reset() {\n')
    f.write('\t\t\tfor (int k=0; k<%d; ++k) {\n' % nx)
    f.write('\t\t\t\t_x[k] = 0.0f;\n')
    f.write('\t\t\t}\n')
    f.write('\t\t}\n')
    f.write('\t\tvoid update(float* u) {\n')
    f.write('\t\t\tfloat x_prev[%d];\n' % nx)
    f.write('\t\t\tfor (int k=0; k<%d; ++k) {\n' % nx)
    f.write('\t\t\t\tx_prev[k] = _x[k];\n')
    f.write('\t\t\t}\n')
    for k in range(nx):
        f.write('\t\t\t_x[%d] =' % k)
        for l in range(nx):
            f.write(' + %f*x_prev[%d]' % (A[k, l], l))
        for l in range(nu):
            f.write(' + %f*u[%d]' % (B[k, l], l))
        f.write(';\n')
    f.write('\t\t}\n')
    f.write('\t\tvoid state(float* x) {\n')
    f.write('\t\t\tfor (int k=0; k<%d; ++k) {\n' % nx)
    f.write('\t\t\t\tx[k] = _x[k];\n')
    f.write('\t\t\t}\n')
    f.write('\t\t}\n')
    f.write('\t\tvoid output(float* u, float* y) {\n')
    for k in range(ny):
        f.write('\t\t\ty[%d] =' % k)
        for l in range(nx):
            f.write(' + %f*_x[%d]' % (C[k, l], l))
        for l in range(nu):
            f.write(' + %f*u[%d]' % (D[k, l], l))
        f.write(';\n')
    f.write('\t\t}\n')
    f.write('\t\tvoid x_ss(float* r, float* x_ss) {\n')
    for k in range(nx):
        f.write('\t\t\tx_ss[%d] =' % k)
        for l in range(1):
            f.write(' + %f*r[%d]' % (Tx[k, l], l))
            f.write(';\n')
    f.write('\t\t}\n')
    f.write('\t\tint nx() {\n')
    f.write('\t\t\treturn %d;\n' % A.shape[0])
    f.write('\t\t}\n')
    f.write('\t\tint nu() {\n')
    f.write('\t\t\treturn %d;\n' % B.shape[1])
    f.write('\t\t}\n')
    f.write('\t\tint ny() {\n')
    f.write('\t\t\treturn %d;\n' % C.shape[0])
    f.write('\t\t}\n')
    f.write('\t\tint nxu() {\n')
    f.write('\t\t\treturn %d;\n' % nxu)
    f.write('\t\t}\n')
    f.write('\t\tdouble Ts() { return _Ts; }\n')
    f.write('\t\tdouble* A() { return _A; }\n')
    f.write('\t\tdouble* B() { return _B; }\n')
    f.write('\t\tdouble* C() { return _C; }\n')
    f.write('\t\tdouble* D() { return _D; }\n')
    f.write('\t\tdouble* Q() { return _Q; }\n')
    f.write('\t\tdouble* R() { return _R; }\n')
    f.write('\t\tdouble* CN() { return _CN; }\n')
    f.write('\t\tdouble* umin() { return _umin; }\n')
    f.write('\t\tdouble* umax() { return _umax; }\n')
    f.write('};\n')
    f.write('#endif\n')
    f.close()


if __name__ == '__main__':
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
    N = 20
    rho = 1.e5
    Q = rho*C.T.dot(C)
    R = np.eye(nq)
    controller = rtPGM(A, B, C, D, Q, R, umin, umax, N, terminal_constraint_tol=1e-8)

    controller.update(np.c_[0,0,0,0].T, 0.12)

    controller.codegen('src/rtPGM.h')
    controller = PGM(A, B, C, D, Q, R, umin, umax, N, terminal_constraint_tol=1e-8, tol=1e-2)
    controller.codegen('src/PGM.h')
    lti_codegen(A, B, CC, DD, Q, R, [umin], [umax], controller.Tx, controller.Su.T, Ts, 'mecotron', 'src/mecotron.h')
