#include "mecotron.h"
#include "PGM.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>

using namespace std;

int main(int argc, char* argv[]) {
    bool verbose = true;
    int n_trials = 1;
    std::string filename = "pgm.csv";

    for (int i=0; i<argc; ++i) {
        std::string arg = argv[i];
        if (((arg == "-v") || (arg == "--verbose")) && (i+1 < argc)) {
            verbose = (std::string(argv[++i]) == "1");
        }
        else if (((arg == "-t") || (arg == "--trials")) && (i+1 < argc)) {
            n_trials = stoi(std::string(argv[++i]));
        }
        else if (((arg == "-f")  || (arg == "--filename")) && (i+1 < argc)) {
            filename = std::string(argv[++i]);
        }
    }

    Mecotron system;
    PGM controller;

    int n_it = 100;
    float Ts = system.Ts();
    float x[system.nx()];
    float y[system.ny()];
    float u[system.nu()];

    float r = 0.12;
    system.state(x);

    // trials
    double ts[n_it];
    for (int it=0; it<n_it; it++) {
        ts[it] = 0.;
    }
    int n_iter;
    ofstream file;
    file.open(filename);
    file << "t,theta,position,pendulum_position,velocity,u,ts,n_it_proj,n_iter\n";
    for (int tr=0; tr<n_trials; tr++) {
        if (verbose) {
            printf("\nTrial %2d/%2d\n", tr+1, n_trials);
            printf("***********\n\n");
        }
        system.reset();
        controller.reset();
        system.state(x);

        // mpc iterations
        int n_it = 100;
        int ret;
        if (verbose) {
            printf("%3s | %8s | %8s \n", "it", "t_solve", "n_iter");
            printf("----|---------|---------\n");
        }
        for (int it=0; it<n_it; it++) {
            auto begin = chrono::high_resolution_clock::now();
            ret = controller.update(x, &r, u);
            auto end = std::chrono::high_resolution_clock::now() - begin;
            long long nanoseconds = chrono::duration_cast<std::chrono::nanoseconds>(end).count();
            double microseconds = static_cast<double>(nanoseconds)/1000.;
            ts[it] += microseconds*1e-6/n_trials;
            n_iter = controller.n_iter();
            if (verbose) {
                printf("%3d | %1.4g ms | %d\n", it, microseconds/1000., n_iter);
            }
            if (n_iter >= 200000) {
                printf("Maximum iterations exceeded!\n");
            }
            system.update(u);
            system.state(x);
            system.output(u, y);
            if (tr == n_trials-1) {
                file << Ts*it << "," << y[0] << "," << y[1] << "," << y[2] << ",";
                file << y[3] << "," << u[0] << "," << ts[it] << ",";
                file << controller.n_it_proj() << "," << n_iter << "\n";
            }
        }
    }
    file.close();
    return 0;
}
