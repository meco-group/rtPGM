#ifndef RTPGM_H
#define RTPGM_H
class rtPGM {
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

	public:
		rtPGM() : _n_it_proj(0) { reset(); }

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
			gradient_step(x, r);
			if (!project(x, r)) {
				return false;
			}
			*u = _q[0];
			shift();
			return true;
		}

};
#endif
