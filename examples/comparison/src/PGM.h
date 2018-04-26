#ifndef PGM_H
#define PGM_H
class PGM {
	private:
		float _q[20];
		int _n_iter;
		int _n_it_proj;

		float fabs(float value) {
			return (value >= 0.0f) ? value : -value;
		}

		float fnorm(float* value) {
			return value[0]*value[0];
		}

		void gradient_step(float* x, float* r) {
			float q_prev[20];
			for (int k=0; k<20; k++) {
				q_prev[k] = _q[k];
			}
			_q[0] = 0.73649751*q_prev[0] + -0.24781073*q_prev[1] + -0.22820379*q_prev[2] + -0.20485252*q_prev[3] + -0.17832235*q_prev[4] + -0.14926634*q_prev[5] + -0.11840183*q_prev[6] + -0.086485469*q_prev[7] + -0.054287607*q_prev[8] + -0.022566877*q_prev[9] + 0.0079541386*q_prev[10] + 0.036611319*q_prev[11] + 0.06281874*q_prev[12] + 0.086085084*q_prev[13] + 0.10602616*q_prev[14] + 0.12237317*q_prev[15] + 0.1349763*q_prev[16] + 0.14380373*q_prev[17] + 0.1489358*q_prev[18] + 0.15055464*q_prev[19] + -13.471524*x[0] + -178.19739*x[1] + -950.40898*x[2] + -4882.8083*x[3] + 6.6065139*r[0];
			_q[1] = -0.24781073*q_prev[0] + 0.76218773*q_prev[1] + -0.22295984*q_prev[2] + -0.20441836*q_prev[3] + -0.182334*q_prev[4] + -0.15724077*q_prev[5] + -0.12975445*q_prev[6] + -0.10055019*q_prev[7] + -0.07033886*q_prev[8] + -0.039842832*q_prev[9] + -0.0097720708*q_prev[10] + 0.019198539*q_prev[11] + 0.046449918*q_prev[12] + 0.07143683*q_prev[13] + 0.093702624*q_prev[14] + 0.11289018*q_prev[15] + 0.12874865*q_prev[16] + 0.14113582*q_prev[17] + 0.15001589*q_prev[18] + 0.15545273*q_prev[19] + -12.573886*x[0] + -155.50698*x[1] + -747.27013*x[2] + -4196.6863*x[3] + 5.6781803*r[0];
			_q[2] = -0.22820379*q_prev[0] + -0.22295984*q_prev[1] + 0.78619929*q_prev[2] + -0.20001374*q_prev[3] + -0.18273917*q_prev[4] + -0.16209175*q_prev[5] + -0.13856821*q_prev[6] + -0.11274213*q_prev[7] + -0.085242904*q_prev[8] + -0.056733413*q_prev[9] + -0.027887354*q_prev[10] + 0.00063300253*q_prev[11] + 0.028197812*q_prev[12] + 0.054228681*q_prev[13] + 0.078215043*q_prev[14] + 0.099727308*q_prev[15] + 0.11842633*q_prev[16] + 0.13406885*q_prev[17] + 0.14650866*q_prev[18] + 0.1556935*q_prev[19] + -11.470795*x[0] + -131.0934*x[1] + -549.3413*x[2] + -3541.5812*x[3] + 4.7918131*r[0];
			_q[3] = -0.20485252*q_prev[0] + -0.20441836*q_prev[1] + -0.20001374*q_prev[2] + 0.80807993*q_prev[3] + -0.17940001*q_prev[4] + -0.16356239*q_prev[5] + -0.14448464*q_prev[6] + -0.12262135*q_prev[7] + -0.098500309*q_prev[8] + -0.072702918*q_prev[9] + -0.045843396*q_prev[10] + -0.018547741*q_prev[11] + 0.0085668138*q_prev[12] + 0.034911114*q_prev[13] + 0.059941433*q_prev[14] + 0.083174266*q_prev[15] + 0.104198*q_prev[16] + 0.12268107*q_prev[17] + 0.13837623*q_prev[18] + 0.15112081*q_prev[19] + -10.188531*x[0] + -105.58494*x[1] + -361.20677*x[2] + -2925.8496*x[3] + 3.958719*r[0];
			_q[4] = -0.17832235*q_prev[0] + -0.182334*q_prev[1] + -0.18273917*q_prev[2] + -0.17940001*q_prev[3] + 0.82742675*q_prev[4] + -0.16149013*q_prev[5] + -0.14722219*q_prev[6] + -0.12980469*q_prev[7] + -0.10964644*q_prev[8] + -0.08722723*q_prev[9] + -0.063079808*q_prev[10] + -0.03777069*q_prev[11] + -0.011880837*q_prev[12] + 0.014013208*q_prev[13] + 0.039356958*q_prev[14] + 0.063633749*q_prev[15] + 0.086378051*q_prev[16] + 0.10718583*q_prev[17] + 0.12572154*q_prev[18] + 0.14172146*q_prev[19] + -8.758473*x[0] + -79.631526*x[1] + -187.06943*x[2] + -2356.936*x[3] + 3.1889702*r[0];
			_q[5] = -0.14926634*q_prev[0] + -0.15724077*q_prev[1] + -0.16209175*q_prev[2] + -0.16356239*q_prev[3] + -0.16149013*q_prev[4] + 0.84389969*q_prev[5] + -0.14658688*q_prev[6] + -0.13397918*q_prev[7] + -0.11826673*q_prev[8] + -0.099810302*q_prev[9] + -0.079041066*q_prev[10] + -0.056444049*q_prev[11] + -0.032540732*q_prev[12] + -0.0078713883*q_prev[13] + 0.017022108*q_prev[14] + 0.041612328*q_prev[15] + 0.065400588*q_prev[16] + 0.087928935*q_prev[17] + 0.10878936*q_prev[18] + 0.12762983*q_prev[19] + -7.2159717*x[0] + -53.88376*x[1] + -30.632194*x[2] + -1841.124*x[3] + 2.4910687*r[0];
			_q[6] = -0.11840183*q_prev[0] + -0.12975445*q_prev[1] + -0.13856821*q_prev[2] + -0.14448464*q_prev[3] + -0.14722219*q_prev[4] + -0.14658688*q_prev[5] + 0.85723326*q_prev[6] + -0.13491355*q_prev[7] + -0.12401089*q_prev[8] + -0.11000027*q_prev[9] + -0.093193816*q_prev[10] + -0.073974985*q_prev[11] + -0.052783769*q_prev[12] + -0.030100962*q_prev[13] + -0.0064321677*q_prev[14] + 0.017707798*q_prev[15] + 0.041809489*q_prev[16] + 0.065381794*q_prev[17] + 0.087962784*q_prev[18] + 0.10912797*q_prev[19] + -5.5991071*x[0] + -28.9725*x[1] + 105.00393*x[2] + -1383.3187*x[3] + 1.8716512*r[0];
			_q[7] = -0.086485469*q_prev[0] + -0.10055019*q_prev[1] + -0.11274213*q_prev[2] + -0.12262135*q_prev[3] + -0.12980469*q_prev[4] + -0.13397918*q_prev[5] + -0.13491355*q_prev[6] + 0.86724634*q_prev[7] + -0.12660551*q_prev[8] + -0.11740469*q_prev[9] + -0.10504403*q_prev[10] + -0.089787984*q_prev[11] + -0.071974954*q_prev[12] + -0.052004247*q_prev[13] + -0.03032199*q_prev[14] + -0.0074067257*q_prev[15] + 0.01624471*q_prev[16] + 0.040130445*q_prev[17] + 0.063755394*q_prev[18] + 0.086641149*q_prev[19] + -3.9473887*x[0] + -5.4894325*x[1] + 217.42216*x[2] + -986.86702*x[3] + 1.3352461*r[0];
			_q[8] = -0.054287607*q_prev[0] + -0.07033886*q_prev[1] + -0.085242904*q_prev[2] + -0.098500309*q_prev[3] + -0.10964644*q_prev[4] + -0.11826673*q_prev[5] + -0.12401089*q_prev[6] + -0.12660551*q_prev[7] + 0.87384934*q_prev[8] + -0.12170434*q_prev[9] + -0.11415349*q_prev[10] + -0.10334324*q_prev[11] + -0.089492991*q_prev[12] + -0.07290047*q_prev[13] + -0.053930313*q_prev[14] + -0.033001586*q_prev[15] + -0.010574852*q_prev[16] + 0.012860629*q_prev[17] + 0.036799008*q_prev[18] + 0.060728721*q_prev[19] + -2.3004374*x[0] + 16.030678*x[1] + 304.95036*x[2] + -653.42418*x[3] + 0.88409284*r[0];
			_q[9] = -0.022566877*q_prev[0] + -0.039842832*q_prev[1] + -0.056733413*q_prev[2] + -0.072702918*q_prev[3] + -0.08722723*q_prev[4] + -0.099810302*q_prev[5] + -0.11000027*q_prev[6] + -0.11740469*q_prev[7] + -0.12170434*q_prev[8] + 0.87704867*q_prev[9] + -0.12015498*q_prev[10] + -0.11415453*q_prev[11] + -0.10475008*q_prev[12] + -0.092120341*q_prev[13] + -0.076528371*q_prev[14] + -0.058311743*q_prev[15] + -0.037871546*q_prev[16] + -0.015660767*q_prev[17] + 0.0078273578*q_prev[18] + 0.0320705*q_prev[19] + -0.69669728*x[0] + 35.125555*x[1] + 366.69797*x[2] + -382.87178*x[3] + 0.51803133*r[0];
			_q[10] = 0.0079541386*q_prev[0] + -0.0097720708*q_prev[1] + -0.027887354*q_prev[2] + -0.045843396*q_prev[3] + -0.063079808*q_prev[4] + -0.079041066*q_prev[5] + -0.093193816*q_prev[6] + -0.10504403*q_prev[7] + -0.11415349*q_prev[8] + -0.12015498*q_prev[9] + 0.87694807*q_prev[10] + -0.12180598*q_prev[11] + -0.11721134*q_prev[12] + -0.1090274*q_prev[13] + -0.09739821*q_prev[14] + -0.082559769*q_prev[15] + -0.064831672*q_prev[16] + -0.04460743*q_prev[17] + -0.022344007*q_prev[18] + 0.0014488817*q_prev[19] + 0.82777981*x[0] + 51.418317*x[1] + 402.5683*x[2] + -173.29313*x[3] + 0.23446824*r[0];
			_q[11] = 0.036611319*q_prev[0] + 0.019198539*q_prev[1] + 0.00063300253*q_prev[2] + -0.018547741*q_prev[3] + -0.03777069*q_prev[4] + -0.056444049*q_prev[5] + -0.073974985*q_prev[6] + -0.089787984*q_prev[7] + -0.10334324*q_prev[8] + -0.11415453*q_prev[9] + -0.12180598*q_prev[10] + 0.87374666*q_prev[11] + -0.1264132*q_prev[12] + -0.12303906*q_prev[13] + -0.11585567*q_prev[14] + -0.10498001*q_prev[15] + -0.090630103*q_prev[16] + -0.073117961*q_prev[17] + -0.052841075*q_prev[18] + -0.030272888*q_prev[19] + 2.2404359*x[0] + 64.627352*x[1] + 413.24625*x[2] + -21.008944*x[3] + 0.02842542*r[0];
			_q[12] = 0.06281874*q_prev[0] + 0.046449918*q_prev[1] + 0.028197812*q_prev[2] + 0.0085668138*q_prev[3] + -0.011880837*q_prev[4] + -0.032540732*q_prev[5] + -0.052783769*q_prev[6] + -0.071974954*q_prev[7] + -0.089492991*q_prev[8] + -0.10475008*q_prev[9] + -0.11721134*q_prev[10] + -0.1264132*q_prev[11] + 0.86773364*q_prev[12] + -0.13364672*q_prev[13] + -0.13127314*q_prev[14] + -0.12484328*q_prev[15] + -0.11445615*q_prev[16] + -0.10032219*q_prev[17] + -0.082757407*q_prev[18] + -0.062175758*q_prev[19] + 3.5131581*x[0] + 74.572883*x[1] + 400.16167*x[2] + 79.324571*x[3] + -0.10732735*r[0];
			_q[13] = 0.086085084*q_prev[0] + 0.07143683*q_prev[1] + 0.054228681*q_prev[2] + 0.034911114*q_prev[3] + 0.014013208*q_prev[4] + -0.0078713883*q_prev[5] + -0.030100962*q_prev[6] + -0.052004247*q_prev[7] + -0.07290047*q_prev[8] + -0.092120341*q_prev[9] + -0.1090274*q_prev[10] + -0.12303906*q_prev[11] + -0.13364672*q_prev[12] + 0.85927961*q_prev[13] + -0.14310131*q_prev[14] + -0.14148125*q_prev[15] + -0.13553992*q_prev[16] + -0.12536875*q_prev[17] + -0.11118215*q_prev[18] + -0.093312599*q_prev[19] + 4.6230479*x[0] + 81.180043*x[1] + 365.4297*x[2] + 134.55477*x[3] + -0.18205465*r[0];
			_q[14] = 0.10602616*q_prev[0] + 0.093702624*q_prev[1] + 0.078215043*q_prev[2] + 0.059941433*q_prev[3] + 0.039356958*q_prev[4] + 0.017022108*q_prev[5] + -0.0064321677*q_prev[6] + -0.03032199*q_prev[7] + -0.053930313*q_prev[8] + -0.076528371*q_prev[9] + -0.09739821*q_prev[10] + -0.11585567*q_prev[11] + -0.13127314*q_prev[12] + -0.14310131*q_prev[13] + 0.84882447*q_prev[14] + -0.15430997*q_prev[15] + -0.15317844*q_prev[16] + -0.14745307*q_prev[17] + -0.13722925*q_prev[18] + -0.12273789*q_prev[19] + 5.5529843*x[0] + 84.47839*x[1] + 311.76984*x[2] + 152.81016*x[3] + -0.20675446*r[0];
			_q[15] = 0.12237317*q_prev[0] + 0.11289018*q_prev[1] + 0.099727308*q_prev[2] + 0.083174266*q_prev[3] + 0.063633749*q_prev[4] + 0.041612328*q_prev[5] + 0.017707798*q_prev[6] + -0.0074067257*q_prev[7] + -0.033001586*q_prev[8] + -0.058311743*q_prev[9] + -0.082559769*q_prev[10] + -0.10498001*q_prev[11] + -0.12484328*q_prev[12] + -0.14148125*q_prev[13] + -0.15430997*q_prev[14] + 0.83686228*q_prev[15] + -0.1667607*q_prev[16] + -0.16584512*q_prev[17] + -0.16006711*q_prev[18] + -0.14953853*q_prev[19] + 6.2919602*x[0] + 84.597903*x[1] + 242.40599*x[2] + 143.21923*x[3] + -0.19377779*r[0];
			_q[16] = 0.1349763*q_prev[0] + 0.12874865*q_prev[1] + 0.11842633*q_prev[2] + 0.104198*q_prev[3] + 0.086378051*q_prev[4] + 0.065400588*q_prev[5] + 0.041809489*q_prev[6] + 0.01624471*q_prev[7] + -0.010574852*q_prev[8] + -0.037871546*q_prev[9] + -0.064831672*q_prev[10] + -0.090630103*q_prev[11] + -0.11445615*q_prev[12] + -0.13553992*q_prev[13] + -0.15317844*q_prev[14] + -0.1667607*q_prev[15] + 0.82392303*q_prev[16] + -0.17991591*q_prev[17] + -0.17894769*q_prev[18] + -0.17286491*q_prev[19] + 6.8351824*x[0] + 81.761586*x[1] + 160.95051*x[2] + 115.57491*x[3] + -0.15637461*r[0];
			_q[17] = 0.14380373*q_prev[0] + 0.14113582*q_prev[1] + 0.13406885*q_prev[2] + 0.12268107*q_prev[3] + 0.10718583*q_prev[4] + 0.087928935*q_prev[5] + 0.065381794*q_prev[6] + 0.040130445*q_prev[7] + 0.012860629*q_prev[8] + -0.015660767*q_prev[9] + -0.04460743*q_prev[10] + -0.073117961*q_prev[11] + -0.10032219*q_prev[12] + -0.12536875*q_prev[13] + -0.14745307*q_prev[14] + -0.16584512*q_prev[15] + -0.17991591*q_prev[16] + 0.81055186*q_prev[17] + -0.19323443*q_prev[18] + -0.19196145*q_prev[19] + 7.1839303*x[0] + 76.274909*x[1] + 71.275176*x[2] + 79.95008*x[3] + -0.10817367*r[0];
			_q[18] = 0.1489358*q_prev[0] + 0.15001589*q_prev[1] + 0.14650866*q_prev[2] + 0.13837623*q_prev[3] + 0.12572154*q_prev[4] + 0.10878936*q_prev[5] + 0.087962784*q_prev[6] + 0.063755394*q_prev[7] + 0.036799008*q_prev[8] + 0.0078273578*q_prev[9] + -0.022344007*q_prev[10] + -0.052841075*q_prev[11] + -0.082757407*q_prev[12] + -0.11118215*q_prev[13] + -0.13722925*q_prev[14] + -0.16006711*q_prev[15] + -0.17894769*q_prev[16] + -0.19323443*q_prev[17] + 0.79728598*q_prev[18] + -0.20619554*q_prev[19] + 7.3451752*x[0] + 68.512412*x[1] + -22.627157*x[2] + 46.270674*x[3] + -0.062604925*r[0];
			_q[19] = 0.15055464*q_prev[0] + 0.15545273*q_prev[1] + 0.1556935*q_prev[2] + 0.15112081*q_prev[3] + 0.14172146*q_prev[4] + 0.12762983*q_prev[5] + 0.10912797*q_prev[6] + 0.086641149*q_prev[7] + 0.060728721*q_prev[8] + 0.0320705*q_prev[9] + 0.0014488817*q_prev[10] + -0.030272888*q_prev[11] + -0.062175758*q_prev[12] + -0.093312599*q_prev[13] + -0.12273789*q_prev[14] + -0.14953853*q_prev[15] + -0.17286491*q_prev[16] + -0.19196145*q_prev[17] + -0.20619554*q_prev[18] + 0.78462963*q_prev[19] + 7.3309703*x[0] + 58.901839*x[1] + -116.78694*x[2] + 23.853661*x[3] + -0.032274365*r[0];
		}

		bool project(float* x, float* r) {
			float b = 0.0013427116*x[0] + 0.015046426*x[1] + 0.12224718*x[2] + 0.99238472*x[3] + -0.0013427116*r[0];
			float lam0 = 0.0f;
			float q[20];
			for (int k=0; k<20; k++) {
				q[k] = _q[k];
			}
			for (int k=0; k<20; k++) {
				if (q[k] < -0.3) {
					q[k] = -0.3;
				} else if (q[k] > 0.3) {
					q[k] = 0.3;
				}
			}
			float f0 = b + 2.6854232e-05*q[0] + 2.6854232e-05*q[1] + 2.6854232e-05*q[2] + 2.6854232e-05*q[3] + 2.6854232e-05*q[4] + 2.6854232e-05*q[5] + 2.6854232e-05*q[6] + 2.6854232e-05*q[7] + 2.6854232e-05*q[8] + 2.6854232e-05*q[9] + 2.6854232e-05*q[10] + 2.6854232e-05*q[11] + 2.6854232e-05*q[12] + 2.6854232e-05*q[13] + 2.6854232e-05*q[14] + 2.6854232e-05*q[15] + 2.6854232e-05*q[16] + 2.6854232e-05*q[17] + 2.6854232e-05*q[18] + 2.6854232e-05*q[19];
			if (fabs(f0) < 1e-09) {
				for (int k=0; k<20; k++) {
					_q[k] = q[k];
				}
				return true;
			}
			float lam1 = 1861.9039*_q[0] + 1861.9039*_q[1] + 1861.9039*_q[2] + 1861.9039*_q[3] + 1861.9039*_q[4] + 1861.9039*_q[5] + 1861.9039*_q[6] + 1861.9039*_q[7] + 1861.9039*_q[8] + 1861.9039*_q[9] + 1861.9039*_q[10] + 1861.9039*_q[11] + 1861.9039*_q[12] + 1861.9039*_q[13] + 1861.9039*_q[14] + 1861.9039*_q[15] + 1861.9039*_q[16] + 1861.9039*_q[17] + 1861.9039*_q[18] + 1861.9039*_q[19] + 69333724*b;
			q[0] = _q[0] - lam1*2.6854232e-05; q[1] = _q[1] - lam1*2.6854232e-05; q[2] = _q[2] - lam1*2.6854232e-05; q[3] = _q[3] - lam1*2.6854232e-05; q[4] = _q[4] - lam1*2.6854232e-05; q[5] = _q[5] - lam1*2.6854232e-05; q[6] = _q[6] - lam1*2.6854232e-05; q[7] = _q[7] - lam1*2.6854232e-05; q[8] = _q[8] - lam1*2.6854232e-05; q[9] = _q[9] - lam1*2.6854232e-05; q[10] = _q[10] - lam1*2.6854232e-05; q[11] = _q[11] - lam1*2.6854232e-05; q[12] = _q[12] - lam1*2.6854232e-05; q[13] = _q[13] - lam1*2.6854232e-05; q[14] = _q[14] - lam1*2.6854232e-05; q[15] = _q[15] - lam1*2.6854232e-05; q[16] = _q[16] - lam1*2.6854232e-05; q[17] = _q[17] - lam1*2.6854232e-05; q[18] = _q[18] - lam1*2.6854232e-05; q[19] = _q[19] - lam1*2.6854232e-05; 
			for (int k=0; k<20; k++) {
				if (q[k] < -0.3) {
					q[k] = -0.3;
				} else if (q[k] > 0.3) {
					q[k] = 0.3;
				}
			}
			float f1 = b + 2.6854232e-05*q[0] + 2.6854232e-05*q[1] + 2.6854232e-05*q[2] + 2.6854232e-05*q[3] + 2.6854232e-05*q[4] + 2.6854232e-05*q[5] + 2.6854232e-05*q[6] + 2.6854232e-05*q[7] + 2.6854232e-05*q[8] + 2.6854232e-05*q[9] + 2.6854232e-05*q[10] + 2.6854232e-05*q[11] + 2.6854232e-05*q[12] + 2.6854232e-05*q[13] + 2.6854232e-05*q[14] + 2.6854232e-05*q[15] + 2.6854232e-05*q[16] + 2.6854232e-05*q[17] + 2.6854232e-05*q[18] + 2.6854232e-05*q[19];
			int cnt = 0;
			float lam2;
			while (fabs(f1) > 1e-09) {
				if (fabs(f1 - f0) < 1e-12) {
					return false;
				}
				lam2 = (lam0*f1 - lam1*f0)/(f1 - f0);
				f0 = f1;
				lam0 = lam1;
				lam1 = lam2;
				q[0] = _q[0] - lam1*2.6854232e-05; q[1] = _q[1] - lam1*2.6854232e-05; q[2] = _q[2] - lam1*2.6854232e-05; q[3] = _q[3] - lam1*2.6854232e-05; q[4] = _q[4] - lam1*2.6854232e-05; q[5] = _q[5] - lam1*2.6854232e-05; q[6] = _q[6] - lam1*2.6854232e-05; q[7] = _q[7] - lam1*2.6854232e-05; q[8] = _q[8] - lam1*2.6854232e-05; q[9] = _q[9] - lam1*2.6854232e-05; q[10] = _q[10] - lam1*2.6854232e-05; q[11] = _q[11] - lam1*2.6854232e-05; q[12] = _q[12] - lam1*2.6854232e-05; q[13] = _q[13] - lam1*2.6854232e-05; q[14] = _q[14] - lam1*2.6854232e-05; q[15] = _q[15] - lam1*2.6854232e-05; q[16] = _q[16] - lam1*2.6854232e-05; q[17] = _q[17] - lam1*2.6854232e-05; q[18] = _q[18] - lam1*2.6854232e-05; q[19] = _q[19] - lam1*2.6854232e-05; 
				for (int k=0; k<20; k++) {
					if (q[k] < -0.3) {
						q[k] = -0.3;
					} else if (q[k] > 0.3) {
						q[k] = 0.3;
					}
				}
				f1 = b + 2.6854232e-05*q[0] + 2.6854232e-05*q[1] + 2.6854232e-05*q[2] + 2.6854232e-05*q[3] + 2.6854232e-05*q[4] + 2.6854232e-05*q[5] + 2.6854232e-05*q[6] + 2.6854232e-05*q[7] + 2.6854232e-05*q[8] + 2.6854232e-05*q[9] + 2.6854232e-05*q[10] + 2.6854232e-05*q[11] + 2.6854232e-05*q[12] + 2.6854232e-05*q[13] + 2.6854232e-05*q[14] + 2.6854232e-05*q[15] + 2.6854232e-05*q[16] + 2.6854232e-05*q[17] + 2.6854232e-05*q[18] + 2.6854232e-05*q[19];
				cnt++;
			}
			for (int k=0; k<20; k++) {
				_q[k] = q[k];
			}
			_n_it_proj = cnt;
			return true;
		}

		void shift() {
			for (int k=0; k<19; k++) {
				_q[k] = _q[k+1];
			}
			_q[19] = 0.0f;
		}

		float residual(float* q1, float* q0) {
			float res = 0;
			float dres;
			dres = -2.5744625*(q1[0] + -q0[0]) + 0.86623431*(q1[1] + -q0[1]) + 0.79769732*(q1[2] + -q0[2]) + 0.71607183*(q1[3] + -q0[3]) + 0.62333435*(q1[4] + -q0[4]) + 0.52176768*(q1[5] + -q0[5]) + 0.4138793*(q1[6] + -q0[6]) + 0.30231412*(q1[7] + -q0[7]) + 0.18976494*(q1[8] + -q0[8]) + 0.078883604*(q1[9] + -q0[9]) + -0.027804074*(q1[10] + -q0[10]) + -0.12797663*(q1[11] + -q0[11]) + -0.21958593*(q1[12] + -q0[12]) + -0.30091455*(q1[13] + -q0[13]) + -0.37061956*(q1[14] + -q0[14]) + -0.42776129*(q1[15] + -q0[15]) + -0.47181614*(q1[16] + -q0[16]) + -0.50267284*(q1[17] + -q0[17]) + -0.52061224*(q1[18] + -q0[18]) + -0.52627097*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 0.86623431*(q1[0] + -q0[0]) + -2.6642639*(q1[1] + -q0[1]) + 0.77936684*(q1[2] + -q0[2]) + 0.71455422*(q1[3] + -q0[3]) + 0.63735726*(q1[4] + -q0[4]) + 0.54964269*(q1[5] + -q0[5]) + 0.45356292*(q1[6] + -q0[6]) + 0.35147801*(q1[7] + -q0[7]) + 0.24587287*(q1[8] + -q0[8]) + 0.13927254*(q1[9] + -q0[9]) + 0.034158744*(q1[10] + -q0[10]) + -0.067109418*(q1[11] + -q0[11]) + -0.16236792*(q1[12] + -q0[12]) + -0.24971088*(q1[13] + -q0[13]) + -0.32754203*(q1[14] + -q0[14]) + -0.39461304*(q1[15] + -q0[15]) + -0.45004709*(q1[16] + -q0[16]) + -0.49334705*(q1[17] + -q0[17]) + -0.52438777*(q1[18] + -q0[18]) + -0.54339249*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 0.79769732*(q1[0] + -q0[0]) + 0.77936684*(q1[1] + -q0[1]) + -2.7481974*(q1[2] + -q0[2]) + 0.69915764*(q1[3] + -q0[3]) + 0.63877356*(q1[4] + -q0[4]) + 0.56659951*(q1[5] + -q0[5]) + 0.48437183*(q1[6] + -q0[6]) + 0.39409554*(q1[7] + -q0[7]) + 0.29797067*(q1[8] + -q0[8]) + 0.19831437*(q1[9] + -q0[9]) + 0.097481586*(q1[10] + -q0[10]) + -0.0022126908*(q1[11] + -q0[11]) + -0.098566807*(q1[12] + -q0[12]) + -0.18955896*(q1[13] + -q0[13]) + -0.27340445*(q1[14] + -q0[14]) + -0.3486016*(q1[15] + -q0[15]) + -0.41396493*(q1[16] + -q0[16]) + -0.4686441*(q1[17] + -q0[17]) + -0.51212808*(q1[18] + -q0[18]) + -0.54423412*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 0.71607183*(q1[0] + -q0[0]) + 0.71455422*(q1[1] + -q0[1]) + 0.69915764*(q1[2] + -q0[2]) + -2.8246823*(q1[3] + -q0[3]) + 0.62710135*(q1[4] + -q0[4]) + 0.57174019*(q1[5] + -q0[5]) + 0.50505302*(q1[6] + -q0[6]) + 0.42862882*(q1[7] + -q0[7]) + 0.34431257*(q1[8] + -q0[8]) + 0.25413655*(q1[9] + -q0[9]) + 0.16024779*(q1[10] + -q0[10]) + 0.064834522*(q1[11] + -q0[11]) + -0.029945709*(q1[12] + -q0[12]) + -0.12203348*(q1[13] + -q0[13]) + -0.20952816*(q1[14] + -q0[14]) + -0.29073965*(q1[15] + -q0[15]) + -0.36422912*(q1[16] + -q0[16]) + -0.42883757*(q1[17] + -q0[17]) + -0.48370075*(q1[18] + -q0[18]) + -0.52825005*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 0.62333435*(q1[0] + -q0[0]) + 0.63735726*(q1[1] + -q0[1]) + 0.63877356*(q1[2] + -q0[2]) + 0.62710135*(q1[3] + -q0[3]) + -2.89231*(q1[4] + -q0[4]) + 0.56449651*(q1[5] + -q0[5]) + 0.51462223*(q1[6] + -q0[6]) + 0.45373855*(q1[7] + -q0[7]) + 0.38327439*(q1[8] + -q0[8]) + 0.30490698*(q1[9] + -q0[9]) + 0.2204985*(q1[10] + -q0[10]) + 0.13202927*(q1[11] + -q0[11]) + 0.041530038*(q1[12] + -q0[12]) + -0.048983844*(q1[13] + -q0[13]) + -0.13757414*(q1[14] + -q0[14]) + -0.22243483*(q1[15] + -q0[15]) + -0.30193863*(q1[16] + -q0[16]) + -0.37467322*(q1[17] + -q0[17]) + -0.43946569*(q1[18] + -q0[18]) + -0.49539419*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 0.52176768*(q1[0] + -q0[0]) + 0.54964269*(q1[1] + -q0[1]) + 0.56659951*(q1[2] + -q0[2]) + 0.57174019*(q1[3] + -q0[3]) + 0.56449651*(q1[4] + -q0[4]) + -2.9498919*(q1[5] + -q0[5]) + 0.51240147*(q1[6] + -q0[6]) + 0.46833067*(q1[7] + -q0[7]) + 0.41340703*(q1[8] + -q0[8]) + 0.34889171*(q1[9] + -q0[9]) + 0.27629185*(q1[10] + -q0[10]) + 0.19730289*(q1[11] + -q0[11]) + 0.11374769*(q1[12] + -q0[12]) + 0.027514816*(q1[13] + -q0[13]) + -0.059501596*(q1[14] + -q0[14]) + -0.14545789*(q1[15] + -q0[15]) + -0.2286109*(q1[16] + -q0[16]) + -0.30735982*(q1[17] + -q0[17]) + -0.38027843*(q1[18] + -q0[18]) + -0.44613621*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 0.4138793*(q1[0] + -q0[0]) + 0.45356292*(q1[1] + -q0[1]) + 0.48437183*(q1[2] + -q0[2]) + 0.50505302*(q1[3] + -q0[3]) + 0.51462223*(q1[4] + -q0[4]) + 0.51240147*(q1[5] + -q0[5]) + -2.9965001*(q1[6] + -q0[6]) + 0.4715968*(q1[7] + -q0[7]) + 0.43348604*(q1[8] + -q0[8]) + 0.38451124*(q1[9] + -q0[9]) + 0.32576347*(q1[10] + -q0[10]) + 0.25858312*(q1[11] + -q0[11]) + 0.1845082*(q1[12] + -q0[12]) + 0.10521936*(q1[13] + -q0[13]) + 0.022483952*(q1[14] + -q0[14]) + -0.061898461*(q1[15] + -q0[15]) + -0.14614708*(q1[16] + -q0[16]) + -0.2285452*(q1[17] + -q0[17]) + -0.30747814*(q1[18] + -q0[18]) + -0.38146207*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 0.30231412*(q1[0] + -q0[0]) + 0.35147801*(q1[1] + -q0[1]) + 0.39409554*(q1[2] + -q0[2]) + 0.42862882*(q1[3] + -q0[3]) + 0.45373855*(q1[4] + -q0[4]) + 0.46833067*(q1[5] + -q0[5]) + 0.4715968*(q1[6] + -q0[6]) + -3.0315013*(q1[7] + -q0[7]) + 0.44255564*(q1[8] + -q0[8]) + 0.41039373*(q1[9] + -q0[9]) + 0.36718647*(q1[10] + -q0[10]) + 0.31385822*(q1[11] + -q0[11]) + 0.25159191*(q1[12] + -q0[12]) + 0.18178335*(q1[13] + -q0[13]) + 0.10599197*(q1[14] + -q0[14]) + 0.025890566*(q1[15] + -q0[15]) + -0.056784166*(q1[16] + -q0[16]) + -0.1402779*(q1[17] + -q0[17]) + -0.22286005*(q1[18] + -q0[18]) + -0.3028583*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 0.18976494*(q1[0] + -q0[0]) + 0.24587287*(q1[1] + -q0[1]) + 0.29797067*(q1[2] + -q0[2]) + 0.34431257*(q1[3] + -q0[3]) + 0.38327439*(q1[4] + -q0[4]) + 0.41340703*(q1[5] + -q0[5]) + 0.43348604*(q1[6] + -q0[6]) + 0.44255564*(q1[7] + -q0[7]) + -3.0545824*(q1[8] + -q0[8]) + 0.42542338*(q1[9] + -q0[9]) + 0.399029*(q1[10] + -q0[10]) + 0.36124126*(q1[11] + -q0[11]) + 0.31282705*(q1[12] + -q0[12]) + 0.2548271*(q1[13] + -q0[13]) + 0.188516*(q1[14] + -q0[14]) + 0.11535863*(q1[15] + -q0[15]) + 0.036964902*(q1[16] + -q0[16]) + -0.044954949*(q1[17] + -q0[17]) + -0.1286327*(q1[18] + -q0[18]) + -0.21228016*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 0.078883604*(q1[0] + -q0[0]) + 0.13927254*(q1[1] + -q0[1]) + 0.19831437*(q1[2] + -q0[2]) + 0.25413655*(q1[3] + -q0[3]) + 0.30490698*(q1[4] + -q0[4]) + 0.34889171*(q1[5] + -q0[5]) + 0.38451124*(q1[6] + -q0[6]) + 0.41039373*(q1[7] + -q0[7]) + 0.42542338*(q1[8] + -q0[8]) + -3.0657658*(q1[9] + -q0[9]) + 0.4200075*(q1[10] + -q0[10]) + 0.39903264*(q1[11] + -q0[11]) + 0.36615895*(q1[12] + -q0[12]) + 0.32201108*(q1[13] + -q0[13]) + 0.2675086*(q1[14] + -q0[14]) + 0.2038315*(q1[15] + -q0[15]) + 0.13238181*(q1[16] + -q0[16]) + 0.054742964*(q1[17] + -q0[17]) + -0.027360906*(q1[18] + -q0[18]) + -0.11210398*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.027804074*(q1[0] + -q0[0]) + 0.034158744*(q1[1] + -q0[1]) + 0.097481586*(q1[2] + -q0[2]) + 0.16024779*(q1[3] + -q0[3]) + 0.2204985*(q1[4] + -q0[4]) + 0.27629185*(q1[5] + -q0[5]) + 0.32576347*(q1[6] + -q0[6]) + 0.36718647*(q1[7] + -q0[7]) + 0.399029*(q1[8] + -q0[8]) + 0.4200075*(q1[9] + -q0[9]) + -3.0654142*(q1[10] + -q0[10]) + 0.42577868*(q1[11] + -q0[11]) + 0.40971786*(q1[12] + -q0[12]) + 0.38111052*(q1[13] + -q0[13]) + 0.34046013*(q1[14] + -q0[14]) + 0.28859164*(q1[15] + -q0[15]) + 0.22662223*(q1[16] + -q0[16]) + 0.15592742*(q1[17] + -q0[17]) + 0.07810455*(q1[18] + -q0[18]) + -0.0050646356*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.12797663*(q1[0] + -q0[0]) + -0.067109418*(q1[1] + -q0[1]) + -0.0022126908*(q1[2] + -q0[2]) + 0.064834522*(q1[3] + -q0[3]) + 0.13202927*(q1[4] + -q0[4]) + 0.19730289*(q1[5] + -q0[5]) + 0.25858312*(q1[6] + -q0[6]) + 0.31385822*(q1[7] + -q0[7]) + 0.36124126*(q1[8] + -q0[8]) + 0.39903264*(q1[9] + -q0[9]) + 0.42577868*(q1[10] + -q0[10]) + -3.0542235*(q1[11] + -q0[11]) + 0.44188342*(q1[12] + -q0[12]) + 0.43008895*(q1[13] + -q0[13]) + 0.40497908*(q1[14] + -q0[14]) + 0.36696269*(q1[15] + -q0[15]) + 0.31680188*(q1[16] + -q0[16]) + 0.25558735*(q1[17] + -q0[17]) + 0.18470852*(q1[18] + -q0[18]) + 0.10582034*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.21958593*(q1[0] + -q0[0]) + -0.16236792*(q1[1] + -q0[1]) + -0.098566807*(q1[2] + -q0[2]) + -0.029945709*(q1[3] + -q0[3]) + 0.041530038*(q1[4] + -q0[4]) + 0.11374769*(q1[5] + -q0[5]) + 0.1845082*(q1[6] + -q0[6]) + 0.25159191*(q1[7] + -q0[7]) + 0.31282705*(q1[8] + -q0[8]) + 0.36615895*(q1[9] + -q0[9]) + 0.40971786*(q1[10] + -q0[10]) + 0.44188342*(q1[11] + -q0[11]) + -3.0332047*(q1[12] + -q0[12]) + 0.46716855*(q1[13] + -q0[13]) + 0.45887156*(q1[14] + -q0[14]) + 0.43639569*(q1[15] + -q0[15]) + 0.40008697*(q1[16] + -q0[16]) + 0.35068105*(q1[17] + -q0[17]) + 0.2892825*(q1[18] + -q0[18]) + 0.21733835*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.30091455*(q1[0] + -q0[0]) + -0.24971088*(q1[1] + -q0[1]) + -0.18955896*(q1[2] + -q0[2]) + -0.12203348*(q1[3] + -q0[3]) + -0.048983844*(q1[4] + -q0[4]) + 0.027514816*(q1[5] + -q0[5]) + 0.10521936*(q1[6] + -q0[6]) + 0.18178335*(q1[7] + -q0[7]) + 0.2548271*(q1[8] + -q0[8]) + 0.32201108*(q1[9] + -q0[9]) + 0.38111052*(q1[10] + -q0[10]) + 0.43008895*(q1[11] + -q0[11]) + 0.46716855*(q1[12] + -q0[12]) + -3.0036532*(q1[13] + -q0[13]) + 0.50021751*(q1[14] + -q0[14]) + 0.49455451*(q1[15] + -q0[15]) + 0.47378632*(q1[16] + -q0[16]) + 0.43823249*(q1[17] + -q0[17]) + 0.38864255*(q1[18] + -q0[18]) + 0.32617868*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.37061956*(q1[0] + -q0[0]) + -0.32754203*(q1[1] + -q0[1]) + -0.27340445*(q1[2] + -q0[2]) + -0.20952816*(q1[3] + -q0[3]) + -0.13757414*(q1[4] + -q0[4]) + -0.059501596*(q1[5] + -q0[5]) + 0.022483952*(q1[6] + -q0[6]) + 0.10599197*(q1[7] + -q0[7]) + 0.188516*(q1[8] + -q0[8]) + 0.2675086*(q1[9] + -q0[9]) + 0.34046013*(q1[10] + -q0[10]) + 0.40497908*(q1[11] + -q0[11]) + 0.45887156*(q1[12] + -q0[12]) + 0.50021751*(q1[13] + -q0[13]) + -2.9671068*(q1[14] + -q0[14]) + 0.53939793*(q1[15] + -q0[15]) + 0.53544262*(q1[16] + -q0[16]) + 0.51542931*(q1[17] + -q0[17]) + 0.47969146*(q1[18] + -q0[18]) + 0.4290362*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.42776129*(q1[0] + -q0[0]) + -0.39461304*(q1[1] + -q0[1]) + -0.3486016*(q1[2] + -q0[2]) + -0.29073965*(q1[3] + -q0[3]) + -0.22243483*(q1[4] + -q0[4]) + -0.14545789*(q1[5] + -q0[5]) + -0.061898461*(q1[6] + -q0[6]) + 0.025890566*(q1[7] + -q0[7]) + 0.11535863*(q1[8] + -q0[8]) + 0.2038315*(q1[9] + -q0[9]) + 0.28859164*(q1[10] + -q0[10]) + 0.36696269*(q1[11] + -q0[11]) + 0.43639569*(q1[12] + -q0[12]) + 0.49455451*(q1[13] + -q0[13]) + 0.53939793*(q1[14] + -q0[14]) + -2.9252923*(q1[15] + -q0[15]) + 0.58292004*(q1[16] + -q0[16]) + 0.5797196*(q1[17] + -q0[17]) + 0.55952227*(q1[18] + -q0[18]) + 0.52271911*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.47181614*(q1[0] + -q0[0]) + -0.45004709*(q1[1] + -q0[1]) + -0.41396493*(q1[2] + -q0[2]) + -0.36422912*(q1[3] + -q0[3]) + -0.30193863*(q1[4] + -q0[4]) + -0.2286109*(q1[5] + -q0[5]) + -0.14614708*(q1[6] + -q0[6]) + -0.056784166*(q1[7] + -q0[7]) + 0.036964902*(q1[8] + -q0[8]) + 0.13238181*(q1[9] + -q0[9]) + 0.22662223*(q1[10] + -q0[10]) + 0.31680188*(q1[11] + -q0[11]) + 0.40008697*(q1[12] + -q0[12]) + 0.47378632*(q1[13] + -q0[13]) + 0.53544262*(q1[14] + -q0[14]) + 0.58292004*(q1[15] + -q0[15]) + -2.8800626*(q1[16] + -q0[16]) + 0.6289047*(q1[17] + -q0[17]) + 0.62552025*(q1[18] + -q0[18]) + 0.6042576*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.50267284*(q1[0] + -q0[0]) + -0.49334705*(q1[1] + -q0[1]) + -0.4686441*(q1[2] + -q0[2]) + -0.42883757*(q1[3] + -q0[3]) + -0.37467322*(q1[4] + -q0[4]) + -0.30735982*(q1[5] + -q0[5]) + -0.2285452*(q1[6] + -q0[6]) + -0.1402779*(q1[7] + -q0[7]) + -0.044954949*(q1[8] + -q0[8]) + 0.054742964*(q1[9] + -q0[9]) + 0.15592742*(q1[10] + -q0[10]) + 0.25558735*(q1[11] + -q0[11]) + 0.35068105*(q1[12] + -q0[12]) + 0.43823249*(q1[13] + -q0[13]) + 0.51542931*(q1[14] + -q0[14]) + 0.5797196*(q1[15] + -q0[15]) + 0.6289047*(q1[16] + -q0[16]) + -2.833323*(q1[17] + -q0[17]) + 0.67546023*(q1[18] + -q0[18]) + 0.67101046*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.52061224*(q1[0] + -q0[0]) + -0.52438777*(q1[1] + -q0[1]) + -0.51212808*(q1[2] + -q0[2]) + -0.48370075*(q1[3] + -q0[3]) + -0.43946569*(q1[4] + -q0[4]) + -0.38027843*(q1[5] + -q0[5]) + -0.30747814*(q1[6] + -q0[6]) + -0.22286005*(q1[7] + -q0[7]) + -0.1286327*(q1[8] + -q0[8]) + -0.027360906*(q1[9] + -q0[9]) + 0.07810455*(q1[10] + -q0[10]) + 0.18470852*(q1[11] + -q0[11]) + 0.2892825*(q1[12] + -q0[12]) + 0.38864255*(q1[13] + -q0[13]) + 0.47969146*(q1[14] + -q0[14]) + 0.55952227*(q1[15] + -q0[15]) + 0.62552025*(q1[16] + -q0[16]) + 0.67546023*(q1[17] + -q0[17]) + -2.7869515*(q1[18] + -q0[18]) + 0.72076642*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -0.52627097*(q1[0] + -q0[0]) + -0.54339249*(q1[1] + -q0[1]) + -0.54423412*(q1[2] + -q0[2]) + -0.52825005*(q1[3] + -q0[3]) + -0.49539419*(q1[4] + -q0[4]) + -0.44613621*(q1[5] + -q0[5]) + -0.38146207*(q1[6] + -q0[6]) + -0.3028583*(q1[7] + -q0[7]) + -0.21228016*(q1[8] + -q0[8]) + -0.11210398*(q1[9] + -q0[9]) + -0.0050646356*(q1[10] + -q0[10]) + 0.10582034*(q1[11] + -q0[11]) + 0.21733835*(q1[12] + -q0[12]) + 0.32617868*(q1[13] + -q0[13]) + 0.4290362*(q1[14] + -q0[14]) + 0.52271911*(q1[15] + -q0[15]) + 0.6042576*(q1[16] + -q0[16]) + 0.67101046*(q1[17] + -q0[17]) + 0.72076642*(q1[18] + -q0[18]) + -2.7427106*(q1[19] + -q0[19]);
			res += dres*dres;
			return res;
		}

	public:
		PGM() : _n_iter(0), _n_it_proj(0) { reset(); }

		void reset() {
			for (int k=0; k<20; k++) {
				_q[k] = 0.0f;
			}
		}

		int N() {
			return 20;
		}

		int n_iter() {
			return _n_iter;
		}

		int n_it_proj() {
			return _n_it_proj;
		}

		void input_trajectory(float* q) {
			for (int k=0; k<20; k++) {
				q[k] = _q[k];
			}
		}

		bool update(float* x, float* r, float* u) {
			_n_iter = 0;
			float q0[20];
			while (true) {
				for (int k=0; k<20; k++) {
					q0[k] = _q[k];
				}
				gradient_step(x, r);
				if (!project(x, r)) {
					return false;
				}
				if (++_n_iter >= 200000) {
					break;
				}
				if (residual(_q, q0) < 1e-08) {
					break;
				}
			}
			*u = _q[0];
			shift();
			return true;
		}

};
#endif
