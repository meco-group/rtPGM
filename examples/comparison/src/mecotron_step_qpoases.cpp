#include "mecotron.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>

#include "blasfeo/include/blasfeo_target.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "blasfeo/include/blasfeo_i_aux_ext_dep.h"

#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/ocp_qp/ocp_qp_condensing_qpoases.h"

#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"


#define N 20

using namespace std;

int main(int argc, char* argv[]) {
    bool verbose = true;
    int n_trials = 1;

    for (int i=0; i<argc; ++i) {
        std::string arg = argv[i];
        if (((arg == "-v") || (arg == "--verbose")) && (i+1 < argc)) {
            verbose = (std::string(argv[++i]) == "1");
        }
        else if (((arg == "-t") || (arg == "--trials")) && (i+1 < argc)) {
            n_trials = stoi(std::string(argv[++i]));
        }
    }

    Mecotron system;

    int n_it = 100;
    float Ts = system.Ts();
    int nx = system.nx();
    int nu = system.nu();
    int ny = system.ny();
    int nxu = system.nxu();
    float x[nx];
    float y[ny];
    float u[nu];
    float x_ss[nx];
    float ref[1] = {0.12};
    system.x_ss(ref, x_ss);

    system.state(x);

    // lti dynamics
    real_t A[nx*nx] = {0};
    real_t B[nx*nu] = {0};
    real_t b[nx] = {0, 0, 0, 0};
    memcpy(A, system.A(), nx*nx*sizeof(double));
    memcpy(B, system.B(), nx*nu*sizeof(double));

    // quadratic objective
    real_t Q[nx*nx] = {0};
    real_t R[nu*nu] = {0};
    real_t S[nu*nx] = {0};
    real_t q[nx] = {0};
    real_t r[nu] = {0};
    memcpy(Q, system.Q(), nx*nx*sizeof(double));
    memcpy(R, system.R(), nu*nu*sizeof(double));

    // bounds
    real_t umin[nu];
    real_t umax[nu];
    real_t xmin[nx] = {-1e8, -1e8, -1e8, -1e8}; // dummy
    real_t xmax[nx] = {1e8, 1e8, 1e8, 1e8};
    memcpy(umin, system.umin(), nu*sizeof(double));
    memcpy(umax, system.umax(), nu*sizeof(double));

    // setup vectors that define problem size
    int_t n_x[N+1];
    int_t n_u[N+1];
    int_t n_b[N+1];
    int_t n_c[N+1];

    for (int k=0; k<N+1; k++) {
        if (k == 0) {
            n_u[k] = nu;
            n_x[k] = nx;
            n_c[k] = 0;
        } else if (k < N) {
            n_u[k] = nu;
            n_x[k] = nx;
            n_c[k] = 0;
        } else {
            n_u[k] = 0;
            n_x[k] = nx;
            n_c[k] = nxu;
        }
        n_b[k] = n_x[k] + n_u[k];
    }
    ocp_qp_in *qp_in = create_ocp_qp_in(N, n_x, n_u, n_b, n_c);
    ocp_qp_out *qp_out = create_ocp_qp_out(N, n_x, n_u, n_b, n_c);
    // copy LTI dynamics and constraints to QP memory
    for (int k=0; k < N+1; k++) {
        ocp_qp_in_copy_objective(Q, S, R, q, r, qp_in, k);
        if (k < N)
            ocp_qp_in_copy_dynamics(A, B, b, qp_in, k);
    }
    int_t **hidxb = (int_t **)qp_in->idxb;
    real_t **hlb = (real_t **)qp_in->lb;
    real_t **hub = (real_t **)qp_in->ub;
    real_t **hlc = (real_t **)qp_in->lc;
    real_t **huc = (real_t **)qp_in->uc;
    real_t **hC = (real_t **)qp_in->Cx;
    // set up bounds
    for (int k=0; k<N+1; k++) {
        for (int i=0; i<n_b[k]; i++) {
            hidxb[k][i] = i;
            if (k == 0) {
                if (i < n_x[k]) {
                    hlb[k][i] = xmin[i];
                    hub[k][i] = xmax[i];
                } else {
                    hlb[k][i] = umin[i - n_x[k]];
                    hub[k][i] = umax[i - n_x[k]];
                }
            } else {

                if (i < n_x[k]) {
                    hlb[k][i] = xmin[i];
                    hub[k][i] = xmax[i];
                } else {
                    hlb[k][i] = umin[i - n_x[k]];
                    hub[k][i] = umax[i - n_x[k]];
                }

                // hlb[k][i] = umin[i];
                // hub[k][i] = umax[i];
            }
        }
    }

    // set up terminal state constraint
    memcpy(hC[N], system.CN(), nx*sizeof(double));


    // trials
    double ts[n_it];
    for (int it=0; it<n_it; it++) {
        ts[it] = 0.;
    }
    ofstream file;
    file.open("qpoases.csv");
    file << "t,theta,position,pendulum_position,velocity,u,ts\n";
    for (int tr=0; tr<n_trials; tr++) {
        if (verbose) {
            printf("\nTrial %2d/%2d\n", tr+1, n_trials);
            printf("***********\n\n");
        }
        // setup solver
        ocp_qp_condensing_qpoases_args *qpoases_args = ocp_qp_condensing_qpoases_create_arguments(qp_in);
        qpoases_args->warm_start = 1; // warm start with dual_sol in memory
        int ws_size = ocp_qp_condensing_qpoases_calculate_workspace_size(qp_in, qpoases_args);
        void *ws = malloc(ws_size);
        int mem_size = ocp_qp_condensing_qpoases_calculate_memory_size(qp_in, qpoases_args);
        void *mem = malloc(mem_size);
        ocp_qp_condensing_qpoases_memory *qpoases_mem;
        ocp_qp_condensing_qpoases_assign_memory(qp_in, qpoases_args, (void **)&qpoases_mem, mem);
        system.reset();
        system.state(x);


        // mpc iterations
        int ret;
        if (verbose) {
            printf("%3s | %8s \n", "it", "t_solve");
            printf("----|---------\n");
        }
        for (int it=0; it<n_it; it++) {
            for (int i=0; i<nx; i++) {
                hlb[0][i] = x[i] - x_ss[i];
                hub[0][i] = x[i] - x_ss[i];
            }
            auto begin = chrono::high_resolution_clock::now();
            ret = ocp_qp_condensing_qpoases(qp_in, qp_out, qpoases_args, qpoases_mem, ws);
            auto end = std::chrono::high_resolution_clock::now() - begin;
            long long nanoseconds = chrono::duration_cast<std::chrono::nanoseconds>(end).count();
            double microseconds = static_cast<double>(nanoseconds)/1000.;
            ts[it] += microseconds/n_trials;
            if (verbose) {
                printf("%3d | %1.4g ms\n", it, microseconds/1000.);
            }
            for (int i=0; i<nu; i++) {
                u[i] = qp_out->u[0][i];
            }
            system.update(u);
            system.state(x);
            system.output(u, y);
            if (tr == n_trials-1) {
                file << Ts*it << "," << y[0] << "," << y[1] << "," << y[2] << ",";
                file << y[3] << "," << u[0] << "," << ts[it] << "\n";
            }
        }
    }
    file.close();
    return 0;
}
