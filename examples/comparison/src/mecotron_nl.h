#ifndef MECOTRON_NL_H
#define MECOTRON_NL_H
#include <math.h>
class Mecotron_nl {
	private:
		double _x[4];
		void update1(float* u) {
			float x_prev[4];
			for (int k=0; k<4; ++k) {
				x_prev[k] = _x[k];
			}
			_x[0] = x_prev[0] + 0.000200*x_prev[1];
			_x[1] = x_prev[1] + (0.002000)*(-x_prev[1] + u[0]);
			_x[2] = x_prev[2] + 0.000200*x_prev[3];
			_x[3] = x_prev[3] + (0.001538)*((cos(x_prev[2])/0.100000)*(-x_prev[1] + u[0]) - 9.810000*sin(x_prev[2]) - 0.195500*x_prev[3]);
		}
	public:
		Mecotron_nl() {
			reset();
		}
		void reset() {
			for (int k=0; k<4; ++k) {
				_x[k] = 0.0f;
			}
		}
		void update(float* u) {
			for (int k=0; k<100; k++) {
				update1(u);
			}
		}
		void state(float* x) {
			x[0] = + 0.000000*_x[0] + 0.000000*_x[1] + 0.000000*_x[2] + 0.012365*_x[3];
			x[1] = + 0.000000*_x[0] + 0.000000*_x[1] + 0.012365*_x[2] + 0.000000*_x[3];
			x[2] = + 0.000000*_x[0] + 0.001353*_x[1] + -0.000246*_x[2] + -0.000162*_x[3];
			x[3] = + 0.001353*_x[0] + -0.000027*_x[1] + -0.000157*_x[2] + 0.000003*_x[3];
		}
		void output(float* u, float* y) {
			y[0] = + 0.000000*_x[0] + 0.000000*_x[1] + 1.000000*_x[2] + 0.000000*_x[3] + 0.000000*u[0];
			y[1] = + 1.000000*_x[0] + 0.000000*_x[1] + 0.000000*_x[2] + -0.000000*_x[3] + 0.000000*u[0];
			y[2] = + 1.000000*_x[0] + 0.000000*_x[1] + -0.150000*_x[2] + -0.000000*_x[3] + 0.000000*u[0];
			y[3] = + 0.000000*_x[0] + 0.000000*_x[1] + 0.000000*_x[2] + 1.000000*_x[3] + 0.000000*u[0];
			y[4] = + 0.000000*_x[0] + 1.000000*_x[1] + 0.000000*_x[2] + 0.000000*_x[3] + 0.000000*u[0];
		}
};
#endif
