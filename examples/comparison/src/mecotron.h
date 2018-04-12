#ifndef MECOTRON_H
#define MECOTRON_H
class Mecotron {
	private:
		double _x[4];
		double _A[16] = {0.782685,0.017804,0.000185,0.000001,-1.757936,0.982200,0.019881,0.000199,-13.158976,-0.136943,0.999069,0.019995,0.000000,0.000000,0.000000,1.000000};
		double _B[4] = {0.017804,0.000185,0.000001,0.000000};
		double _C[20] = {0.000000,0.000000,0.000000,80.876000,9.684800,80.876000,9.684800,-2.446600,0.000000,14.731000,0.000000,14.731000,14.731000,0.000000,739.090000,0.000000,739.090000,739.090000,0.000000,0.000000};
		double _D[5] = {0.000000,0.000000,0.000000,0.000000,0.000000};
		double _Q[16] = {0.000000,0.000000,0.000000,0.000000,0.000000,598585.156000,-3604086.460000,-180825759.400000,0.000000,-3604086.460000,21700236.100000,1088753479.000000,0.000000,-180825759.400000,1088753479.000000,54625402810.000000};
		double _R[1] = {1.000000};
		double _CN[4] = {0.001343,0.015046,0.122247,0.992385};
		double _umin[1] = {-0.300000};
		double _umax[1] = {0.300000};
		double _Ts = 0.020000;
	public:
		Mecotron() {
			reset();
		}
		void reset() {
			for (int k=0; k<4; ++k) {
				_x[k] = 0.0f;
			}
		}
		void update(float* u) {
			float x_prev[4];
			for (int k=0; k<4; ++k) {
				x_prev[k] = _x[k];
			}
			_x[0] = + 0.782685*x_prev[0] + -1.757936*x_prev[1] + -13.158976*x_prev[2] + 0.000000*x_prev[3] + 0.017804*u[0];
			_x[1] = + 0.017804*x_prev[0] + 0.982200*x_prev[1] + -0.136943*x_prev[2] + 0.000000*x_prev[3] + 0.000185*u[0];
			_x[2] = + 0.000185*x_prev[0] + 0.019881*x_prev[1] + 0.999069*x_prev[2] + 0.000000*x_prev[3] + 0.000001*u[0];
			_x[3] = + 0.000001*x_prev[0] + 0.000199*x_prev[1] + 0.019995*x_prev[2] + 1.000000*x_prev[3] + 0.000000*u[0];
		}
		void state(float* x) {
			for (int k=0; k<4; ++k) {
				x[k] = _x[k];
			}
		}
		void output(float* u, float* y) {
			y[0] = + 0.000000*_x[0] + 80.876000*_x[1] + 0.000000*_x[2] + 0.000000*_x[3] + 0.000000*u[0];
			y[1] = + 0.000000*_x[0] + 9.684800*_x[1] + 14.731000*_x[2] + 739.090000*_x[3] + 0.000000*u[0];
			y[2] = + 0.000000*_x[0] + -2.446600*_x[1] + 14.731000*_x[2] + 739.090000*_x[3] + 0.000000*u[0];
			y[3] = + 80.876000*_x[0] + 0.000000*_x[1] + 0.000000*_x[2] + 0.000000*_x[3] + 0.000000*u[0];
			y[4] = + 9.684800*_x[0] + 14.731000*_x[1] + 739.090000*_x[2] + 0.000000*_x[3] + 0.000000*u[0];
		}
		void x_ss(float* r, float* x_ss) {
			x_ss[0] = + 0.000000*r[0];
			x_ss[1] = + 0.000000*r[0];
			x_ss[2] = + 0.000000*r[0];
			x_ss[3] = + 0.001353*r[0];
		}
		int nx() {
			return 4;
		}
		int nu() {
			return 1;
		}
		int ny() {
			return 5;
		}
		int nxu() {
			return 1;
		}
		double Ts() { return _Ts; }
		double* A() { return _A; }
		double* B() { return _B; }
		double* C() { return _C; }
		double* D() { return _D; }
		double* Q() { return _Q; }
		double* R() { return _R; }
		double* CN() { return _CN; }
		double* umin() { return _umin; }
		double* umax() { return _umax; }
};
#endif
