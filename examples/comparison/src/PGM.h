#ifndef PGM_H
#define PGM_H
class PGM {
	private:
		float _q[20];
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
			if (fabs(f0) < 1e-08) {
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
			while (fabs(f1) > 1e-08) {
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
			dres = -2574.4625*(q1[0] + -q0[0]) + 866.23431*(q1[1] + -q0[1]) + 797.69732*(q1[2] + -q0[2]) + 716.07183*(q1[3] + -q0[3]) + 623.33435*(q1[4] + -q0[4]) + 521.76768*(q1[5] + -q0[5]) + 413.8793*(q1[6] + -q0[6]) + 302.31412*(q1[7] + -q0[7]) + 189.76494*(q1[8] + -q0[8]) + 78.883604*(q1[9] + -q0[9]) + -27.804074*(q1[10] + -q0[10]) + -127.97663*(q1[11] + -q0[11]) + -219.58593*(q1[12] + -q0[12]) + -300.91455*(q1[13] + -q0[13]) + -370.61956*(q1[14] + -q0[14]) + -427.76129*(q1[15] + -q0[15]) + -471.81614*(q1[16] + -q0[16]) + -502.67284*(q1[17] + -q0[17]) + -520.61224*(q1[18] + -q0[18]) + -526.27097*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 866.23431*(q1[0] + -q0[0]) + -2664.2639*(q1[1] + -q0[1]) + 779.36684*(q1[2] + -q0[2]) + 714.55422*(q1[3] + -q0[3]) + 637.35726*(q1[4] + -q0[4]) + 549.64269*(q1[5] + -q0[5]) + 453.56292*(q1[6] + -q0[6]) + 351.47801*(q1[7] + -q0[7]) + 245.87287*(q1[8] + -q0[8]) + 139.27254*(q1[9] + -q0[9]) + 34.158744*(q1[10] + -q0[10]) + -67.109418*(q1[11] + -q0[11]) + -162.36792*(q1[12] + -q0[12]) + -249.71088*(q1[13] + -q0[13]) + -327.54203*(q1[14] + -q0[14]) + -394.61304*(q1[15] + -q0[15]) + -450.04709*(q1[16] + -q0[16]) + -493.34705*(q1[17] + -q0[17]) + -524.38777*(q1[18] + -q0[18]) + -543.39249*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 797.69732*(q1[0] + -q0[0]) + 779.36684*(q1[1] + -q0[1]) + -2748.1974*(q1[2] + -q0[2]) + 699.15764*(q1[3] + -q0[3]) + 638.77356*(q1[4] + -q0[4]) + 566.59951*(q1[5] + -q0[5]) + 484.37183*(q1[6] + -q0[6]) + 394.09554*(q1[7] + -q0[7]) + 297.97067*(q1[8] + -q0[8]) + 198.31437*(q1[9] + -q0[9]) + 97.481586*(q1[10] + -q0[10]) + -2.2126908*(q1[11] + -q0[11]) + -98.566807*(q1[12] + -q0[12]) + -189.55896*(q1[13] + -q0[13]) + -273.40445*(q1[14] + -q0[14]) + -348.6016*(q1[15] + -q0[15]) + -413.96493*(q1[16] + -q0[16]) + -468.6441*(q1[17] + -q0[17]) + -512.12808*(q1[18] + -q0[18]) + -544.23412*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 716.07183*(q1[0] + -q0[0]) + 714.55422*(q1[1] + -q0[1]) + 699.15764*(q1[2] + -q0[2]) + -2824.6823*(q1[3] + -q0[3]) + 627.10135*(q1[4] + -q0[4]) + 571.74019*(q1[5] + -q0[5]) + 505.05302*(q1[6] + -q0[6]) + 428.62882*(q1[7] + -q0[7]) + 344.31257*(q1[8] + -q0[8]) + 254.13655*(q1[9] + -q0[9]) + 160.24779*(q1[10] + -q0[10]) + 64.834522*(q1[11] + -q0[11]) + -29.945709*(q1[12] + -q0[12]) + -122.03348*(q1[13] + -q0[13]) + -209.52816*(q1[14] + -q0[14]) + -290.73965*(q1[15] + -q0[15]) + -364.22912*(q1[16] + -q0[16]) + -428.83757*(q1[17] + -q0[17]) + -483.70075*(q1[18] + -q0[18]) + -528.25005*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 623.33435*(q1[0] + -q0[0]) + 637.35726*(q1[1] + -q0[1]) + 638.77356*(q1[2] + -q0[2]) + 627.10135*(q1[3] + -q0[3]) + -2892.31*(q1[4] + -q0[4]) + 564.49651*(q1[5] + -q0[5]) + 514.62223*(q1[6] + -q0[6]) + 453.73855*(q1[7] + -q0[7]) + 383.27439*(q1[8] + -q0[8]) + 304.90698*(q1[9] + -q0[9]) + 220.4985*(q1[10] + -q0[10]) + 132.02927*(q1[11] + -q0[11]) + 41.530038*(q1[12] + -q0[12]) + -48.983844*(q1[13] + -q0[13]) + -137.57414*(q1[14] + -q0[14]) + -222.43483*(q1[15] + -q0[15]) + -301.93863*(q1[16] + -q0[16]) + -374.67322*(q1[17] + -q0[17]) + -439.46569*(q1[18] + -q0[18]) + -495.39419*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 521.76768*(q1[0] + -q0[0]) + 549.64269*(q1[1] + -q0[1]) + 566.59951*(q1[2] + -q0[2]) + 571.74019*(q1[3] + -q0[3]) + 564.49651*(q1[4] + -q0[4]) + -2949.8919*(q1[5] + -q0[5]) + 512.40147*(q1[6] + -q0[6]) + 468.33067*(q1[7] + -q0[7]) + 413.40703*(q1[8] + -q0[8]) + 348.89171*(q1[9] + -q0[9]) + 276.29185*(q1[10] + -q0[10]) + 197.30289*(q1[11] + -q0[11]) + 113.74769*(q1[12] + -q0[12]) + 27.514816*(q1[13] + -q0[13]) + -59.501596*(q1[14] + -q0[14]) + -145.45789*(q1[15] + -q0[15]) + -228.6109*(q1[16] + -q0[16]) + -307.35982*(q1[17] + -q0[17]) + -380.27843*(q1[18] + -q0[18]) + -446.13621*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 413.8793*(q1[0] + -q0[0]) + 453.56292*(q1[1] + -q0[1]) + 484.37183*(q1[2] + -q0[2]) + 505.05302*(q1[3] + -q0[3]) + 514.62223*(q1[4] + -q0[4]) + 512.40147*(q1[5] + -q0[5]) + -2996.5001*(q1[6] + -q0[6]) + 471.5968*(q1[7] + -q0[7]) + 433.48604*(q1[8] + -q0[8]) + 384.51124*(q1[9] + -q0[9]) + 325.76347*(q1[10] + -q0[10]) + 258.58312*(q1[11] + -q0[11]) + 184.5082*(q1[12] + -q0[12]) + 105.21936*(q1[13] + -q0[13]) + 22.483952*(q1[14] + -q0[14]) + -61.898461*(q1[15] + -q0[15]) + -146.14708*(q1[16] + -q0[16]) + -228.5452*(q1[17] + -q0[17]) + -307.47814*(q1[18] + -q0[18]) + -381.46207*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 302.31412*(q1[0] + -q0[0]) + 351.47801*(q1[1] + -q0[1]) + 394.09554*(q1[2] + -q0[2]) + 428.62882*(q1[3] + -q0[3]) + 453.73855*(q1[4] + -q0[4]) + 468.33067*(q1[5] + -q0[5]) + 471.5968*(q1[6] + -q0[6]) + -3031.5013*(q1[7] + -q0[7]) + 442.55564*(q1[8] + -q0[8]) + 410.39373*(q1[9] + -q0[9]) + 367.18647*(q1[10] + -q0[10]) + 313.85822*(q1[11] + -q0[11]) + 251.59191*(q1[12] + -q0[12]) + 181.78335*(q1[13] + -q0[13]) + 105.99197*(q1[14] + -q0[14]) + 25.890566*(q1[15] + -q0[15]) + -56.784166*(q1[16] + -q0[16]) + -140.2779*(q1[17] + -q0[17]) + -222.86005*(q1[18] + -q0[18]) + -302.8583*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 189.76494*(q1[0] + -q0[0]) + 245.87287*(q1[1] + -q0[1]) + 297.97067*(q1[2] + -q0[2]) + 344.31257*(q1[3] + -q0[3]) + 383.27439*(q1[4] + -q0[4]) + 413.40703*(q1[5] + -q0[5]) + 433.48604*(q1[6] + -q0[6]) + 442.55564*(q1[7] + -q0[7]) + -3054.5824*(q1[8] + -q0[8]) + 425.42338*(q1[9] + -q0[9]) + 399.029*(q1[10] + -q0[10]) + 361.24126*(q1[11] + -q0[11]) + 312.82705*(q1[12] + -q0[12]) + 254.8271*(q1[13] + -q0[13]) + 188.516*(q1[14] + -q0[14]) + 115.35863*(q1[15] + -q0[15]) + 36.964902*(q1[16] + -q0[16]) + -44.954949*(q1[17] + -q0[17]) + -128.6327*(q1[18] + -q0[18]) + -212.28016*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = 78.883604*(q1[0] + -q0[0]) + 139.27254*(q1[1] + -q0[1]) + 198.31437*(q1[2] + -q0[2]) + 254.13655*(q1[3] + -q0[3]) + 304.90698*(q1[4] + -q0[4]) + 348.89171*(q1[5] + -q0[5]) + 384.51124*(q1[6] + -q0[6]) + 410.39373*(q1[7] + -q0[7]) + 425.42338*(q1[8] + -q0[8]) + -3065.7658*(q1[9] + -q0[9]) + 420.0075*(q1[10] + -q0[10]) + 399.03264*(q1[11] + -q0[11]) + 366.15895*(q1[12] + -q0[12]) + 322.01108*(q1[13] + -q0[13]) + 267.5086*(q1[14] + -q0[14]) + 203.8315*(q1[15] + -q0[15]) + 132.38181*(q1[16] + -q0[16]) + 54.742964*(q1[17] + -q0[17]) + -27.360906*(q1[18] + -q0[18]) + -112.10398*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -27.804074*(q1[0] + -q0[0]) + 34.158744*(q1[1] + -q0[1]) + 97.481586*(q1[2] + -q0[2]) + 160.24779*(q1[3] + -q0[3]) + 220.4985*(q1[4] + -q0[4]) + 276.29185*(q1[5] + -q0[5]) + 325.76347*(q1[6] + -q0[6]) + 367.18647*(q1[7] + -q0[7]) + 399.029*(q1[8] + -q0[8]) + 420.0075*(q1[9] + -q0[9]) + -3065.4142*(q1[10] + -q0[10]) + 425.77868*(q1[11] + -q0[11]) + 409.71786*(q1[12] + -q0[12]) + 381.11052*(q1[13] + -q0[13]) + 340.46013*(q1[14] + -q0[14]) + 288.59164*(q1[15] + -q0[15]) + 226.62223*(q1[16] + -q0[16]) + 155.92742*(q1[17] + -q0[17]) + 78.10455*(q1[18] + -q0[18]) + -5.0646356*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -127.97663*(q1[0] + -q0[0]) + -67.109418*(q1[1] + -q0[1]) + -2.2126908*(q1[2] + -q0[2]) + 64.834522*(q1[3] + -q0[3]) + 132.02927*(q1[4] + -q0[4]) + 197.30289*(q1[5] + -q0[5]) + 258.58312*(q1[6] + -q0[6]) + 313.85822*(q1[7] + -q0[7]) + 361.24126*(q1[8] + -q0[8]) + 399.03264*(q1[9] + -q0[9]) + 425.77868*(q1[10] + -q0[10]) + -3054.2235*(q1[11] + -q0[11]) + 441.88342*(q1[12] + -q0[12]) + 430.08895*(q1[13] + -q0[13]) + 404.97908*(q1[14] + -q0[14]) + 366.96269*(q1[15] + -q0[15]) + 316.80188*(q1[16] + -q0[16]) + 255.58735*(q1[17] + -q0[17]) + 184.70852*(q1[18] + -q0[18]) + 105.82034*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -219.58593*(q1[0] + -q0[0]) + -162.36792*(q1[1] + -q0[1]) + -98.566807*(q1[2] + -q0[2]) + -29.945709*(q1[3] + -q0[3]) + 41.530038*(q1[4] + -q0[4]) + 113.74769*(q1[5] + -q0[5]) + 184.5082*(q1[6] + -q0[6]) + 251.59191*(q1[7] + -q0[7]) + 312.82705*(q1[8] + -q0[8]) + 366.15895*(q1[9] + -q0[9]) + 409.71786*(q1[10] + -q0[10]) + 441.88342*(q1[11] + -q0[11]) + -3033.2047*(q1[12] + -q0[12]) + 467.16855*(q1[13] + -q0[13]) + 458.87156*(q1[14] + -q0[14]) + 436.39569*(q1[15] + -q0[15]) + 400.08697*(q1[16] + -q0[16]) + 350.68105*(q1[17] + -q0[17]) + 289.2825*(q1[18] + -q0[18]) + 217.33835*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -300.91455*(q1[0] + -q0[0]) + -249.71088*(q1[1] + -q0[1]) + -189.55896*(q1[2] + -q0[2]) + -122.03348*(q1[3] + -q0[3]) + -48.983844*(q1[4] + -q0[4]) + 27.514816*(q1[5] + -q0[5]) + 105.21936*(q1[6] + -q0[6]) + 181.78335*(q1[7] + -q0[7]) + 254.8271*(q1[8] + -q0[8]) + 322.01108*(q1[9] + -q0[9]) + 381.11052*(q1[10] + -q0[10]) + 430.08895*(q1[11] + -q0[11]) + 467.16855*(q1[12] + -q0[12]) + -3003.6532*(q1[13] + -q0[13]) + 500.21751*(q1[14] + -q0[14]) + 494.55451*(q1[15] + -q0[15]) + 473.78632*(q1[16] + -q0[16]) + 438.23249*(q1[17] + -q0[17]) + 388.64255*(q1[18] + -q0[18]) + 326.17868*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -370.61956*(q1[0] + -q0[0]) + -327.54203*(q1[1] + -q0[1]) + -273.40445*(q1[2] + -q0[2]) + -209.52816*(q1[3] + -q0[3]) + -137.57414*(q1[4] + -q0[4]) + -59.501596*(q1[5] + -q0[5]) + 22.483952*(q1[6] + -q0[6]) + 105.99197*(q1[7] + -q0[7]) + 188.516*(q1[8] + -q0[8]) + 267.5086*(q1[9] + -q0[9]) + 340.46013*(q1[10] + -q0[10]) + 404.97908*(q1[11] + -q0[11]) + 458.87156*(q1[12] + -q0[12]) + 500.21751*(q1[13] + -q0[13]) + -2967.1068*(q1[14] + -q0[14]) + 539.39793*(q1[15] + -q0[15]) + 535.44262*(q1[16] + -q0[16]) + 515.42931*(q1[17] + -q0[17]) + 479.69146*(q1[18] + -q0[18]) + 429.0362*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -427.76129*(q1[0] + -q0[0]) + -394.61304*(q1[1] + -q0[1]) + -348.6016*(q1[2] + -q0[2]) + -290.73965*(q1[3] + -q0[3]) + -222.43483*(q1[4] + -q0[4]) + -145.45789*(q1[5] + -q0[5]) + -61.898461*(q1[6] + -q0[6]) + 25.890566*(q1[7] + -q0[7]) + 115.35863*(q1[8] + -q0[8]) + 203.8315*(q1[9] + -q0[9]) + 288.59164*(q1[10] + -q0[10]) + 366.96269*(q1[11] + -q0[11]) + 436.39569*(q1[12] + -q0[12]) + 494.55451*(q1[13] + -q0[13]) + 539.39793*(q1[14] + -q0[14]) + -2925.2923*(q1[15] + -q0[15]) + 582.92004*(q1[16] + -q0[16]) + 579.7196*(q1[17] + -q0[17]) + 559.52227*(q1[18] + -q0[18]) + 522.71911*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -471.81614*(q1[0] + -q0[0]) + -450.04709*(q1[1] + -q0[1]) + -413.96493*(q1[2] + -q0[2]) + -364.22912*(q1[3] + -q0[3]) + -301.93863*(q1[4] + -q0[4]) + -228.6109*(q1[5] + -q0[5]) + -146.14708*(q1[6] + -q0[6]) + -56.784166*(q1[7] + -q0[7]) + 36.964902*(q1[8] + -q0[8]) + 132.38181*(q1[9] + -q0[9]) + 226.62223*(q1[10] + -q0[10]) + 316.80188*(q1[11] + -q0[11]) + 400.08697*(q1[12] + -q0[12]) + 473.78632*(q1[13] + -q0[13]) + 535.44262*(q1[14] + -q0[14]) + 582.92004*(q1[15] + -q0[15]) + -2880.0626*(q1[16] + -q0[16]) + 628.9047*(q1[17] + -q0[17]) + 625.52025*(q1[18] + -q0[18]) + 604.2576*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -502.67284*(q1[0] + -q0[0]) + -493.34705*(q1[1] + -q0[1]) + -468.6441*(q1[2] + -q0[2]) + -428.83757*(q1[3] + -q0[3]) + -374.67322*(q1[4] + -q0[4]) + -307.35982*(q1[5] + -q0[5]) + -228.5452*(q1[6] + -q0[6]) + -140.2779*(q1[7] + -q0[7]) + -44.954949*(q1[8] + -q0[8]) + 54.742964*(q1[9] + -q0[9]) + 155.92742*(q1[10] + -q0[10]) + 255.58735*(q1[11] + -q0[11]) + 350.68105*(q1[12] + -q0[12]) + 438.23249*(q1[13] + -q0[13]) + 515.42931*(q1[14] + -q0[14]) + 579.7196*(q1[15] + -q0[15]) + 628.9047*(q1[16] + -q0[16]) + -2833.323*(q1[17] + -q0[17]) + 675.46023*(q1[18] + -q0[18]) + 671.01046*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -520.61224*(q1[0] + -q0[0]) + -524.38777*(q1[1] + -q0[1]) + -512.12808*(q1[2] + -q0[2]) + -483.70075*(q1[3] + -q0[3]) + -439.46569*(q1[4] + -q0[4]) + -380.27843*(q1[5] + -q0[5]) + -307.47814*(q1[6] + -q0[6]) + -222.86005*(q1[7] + -q0[7]) + -128.6327*(q1[8] + -q0[8]) + -27.360906*(q1[9] + -q0[9]) + 78.10455*(q1[10] + -q0[10]) + 184.70852*(q1[11] + -q0[11]) + 289.2825*(q1[12] + -q0[12]) + 388.64255*(q1[13] + -q0[13]) + 479.69146*(q1[14] + -q0[14]) + 559.52227*(q1[15] + -q0[15]) + 625.52025*(q1[16] + -q0[16]) + 675.46023*(q1[17] + -q0[17]) + -2786.9515*(q1[18] + -q0[18]) + 720.76642*(q1[19] + -q0[19]);
			res += dres*dres;
			dres = -526.27097*(q1[0] + -q0[0]) + -543.39249*(q1[1] + -q0[1]) + -544.23412*(q1[2] + -q0[2]) + -528.25005*(q1[3] + -q0[3]) + -495.39419*(q1[4] + -q0[4]) + -446.13621*(q1[5] + -q0[5]) + -381.46207*(q1[6] + -q0[6]) + -302.8583*(q1[7] + -q0[7]) + -212.28016*(q1[8] + -q0[8]) + -112.10398*(q1[9] + -q0[9]) + -5.0646356*(q1[10] + -q0[10]) + 105.82034*(q1[11] + -q0[11]) + 217.33835*(q1[12] + -q0[12]) + 326.17868*(q1[13] + -q0[13]) + 429.0362*(q1[14] + -q0[14]) + 522.71911*(q1[15] + -q0[15]) + 604.2576*(q1[16] + -q0[16]) + 671.01046*(q1[17] + -q0[17]) + 720.76642*(q1[18] + -q0[18]) + -2742.7106*(q1[19] + -q0[19]);
			res += dres*dres;
			return res;
		}

	public:
		PGM() : _n_it_proj(0) { reset(); }

		void reset() {
			for (int k=0; k<20; k++) {
				_q[k] = 0.0f;
			}
		}

		int N() {
			return 20;
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
			int cnt = 0;
			float q0[20];
			while (true) {
				for (int k=0; k<20; k++) {
					q0[k] = _q[k];
				}
				gradient_step(x, r);
				if (!project(x, r)) {
					return false;
				}
				if (++cnt > 1000) {
					break;
				}
				if (residual(_q, q0) < 0.0001) {
					break;
				}
			}
			*u = _q[0];
			shift();
			return true;
		}

};
#endif
