#ifndef LQR_H
#define LQR_H
class LQR {
	private:
	public:
		LQR() { reset(); }

		void reset() {
		}

		bool update(float* x, float* r, float* u) {
			*u = -29.723915*x[0] + -793.7349*x[1] + -9840.5372*x[2] + -51989.026*x[3] + 70.341943*r[0];
			return true;
		}

};
#endif
