#include "mecotron.h"
#include "mecotron_nl.h"
#include "rtPGM.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string.h>

using namespace std;

int main(int argc, char* argv[]) {
    bool verbose = true;
    int n_trials = 1;
    std::string filename = "rtpgm.csv";
    bool nonlinear = true;

    for (int i=0; i<argc; ++i) {
        std::string arg = argv[i];
        if (((arg == "-v") || (arg == "--verbose")) && (i+1 < argc)) {
            verbose = (std::string(argv[++i]) == "1");
            continue;
        }
        else if (((arg == "-n") || (arg == "--nonlinear")) && (i+1 < argc)) {
            nonlinear = (std::string(argv[++i]) == "1");
            continue;
        }
        else if (((arg == "-t") || (arg == "--trials")) && (i+1 < argc)) {
            n_trials = stoi(std::string(argv[++i]));
            continue;
        }
        else if (((arg == "-f") || (arg == "--filename")) && (i+1 < argc)) {
            filename = std::string(argv[++i]);
            continue;
        }
    }

    std::cout << "verbose: " << verbose << std::endl;
    std::cout << "nonlinear: " << nonlinear << std::endl;
    std::cout << "filename: " << filename << std::endl;
    std::cout << "n_trials: " << n_trials << std::endl;

    Mecotron system;
    Mecotron_nl system_nl;
    rtPGM controller;

    int n_it = 100;
    float Ts = system.Ts();
    float x[system.nx()];
    float y[system.ny()];
    float u[system.nu()];

    long long times[3];

    float r = 0.12;

    // trials
    double ts[n_it];
    for (int it=0; it<n_it; it++) {
        ts[it] = 0.;
    }
    ofstream file;
    file.open(filename);
    file << "t,theta,position,pendulum_position,velocity,u,ts,n_it_proj,grad_time,proj_time\n";
    for (int tr=0; tr<n_trials; tr++) {
        if (verbose) {
            printf("\nTrial %2d/%2d\n", tr+1, n_trials);
            printf("***********\n\n");
        }
        if (nonlinear) {
            system_nl.reset();
            system_nl.state(x);
        } else {
            system.reset();
            system.state(x);
        }
        controller.reset();
        // mpc iterations
        int ret;
        if (verbose) {
            printf("%3s | %8s \n", "it", "t_solve");
            printf("----|---------\n");
        }
        for (int it=0; it<n_it; it++) {
            auto begin = chrono::high_resolution_clock::now();
            ret = controller.update(x, &r, u);
            auto end = std::chrono::high_resolution_clock::now() - begin;
            long long nanoseconds = chrono::duration_cast<std::chrono::nanoseconds>(end).count();
            double microseconds = static_cast<double>(nanoseconds)/1000.;
            ts[it] += microseconds*1e-6/n_trials;
            if (verbose) {
                printf("%3d | %1.4g ms\n", it, microseconds/1000.);
            }
            if (nonlinear) {
                system_nl.update(u);
                system_nl.state(x);
                system_nl.output(u, y);
            } else {
                system.update(u);
                system.state(x);
                system.output(u, y);
            }
            controller.time_analysis(times);
            double t0 = static_cast<double>(times[0]);
            double t1 = static_cast<double>(times[1]);
            double t2 = static_cast<double>(times[2]);
            // printf("%f\n", t2);
            double t_sum = t0+t1+t2;
            printf("%d - %f - %f\n", controller.n_it_proj(), t0/t_sum, t1/t_sum);
            // printf("%f - %f - %f - %f\n", t0, t1, t2, t0+t1+t2);
            if (tr == n_trials-1) {
                file << Ts*it << "," << y[0] << "," << y[1] << "," << y[2] << ",";
                file << y[3] << "," << u[0] << "," << ts[it] << ",";
                file << controller.n_it_proj() << ",";
                file << t0/t_sum << "," << t1/t_sum << "\n";
            }
        }
    }
    file.close();
    return 0;
}
