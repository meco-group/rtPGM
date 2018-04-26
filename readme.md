# rtPGM -- real-time PGM for linear MPC
rtPGM is a compact Python toolbox for the design, simulation and code generation of the real-time PGM algorithm. The real-time PGM (Proximal Gradient Method) is an elegant, fast and computational cheap MPC (Model Predictive Control) algorithm for controlling linear systems with hard input bounds and allows to obtain fast control rates on embedded hardware.

## Usage
The real-time PGM is a linear MPC approach that examines the following optimal control problem:
<center>
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;&&space;\underset{q,\,s}{\text{minimize}}&space;&&space;&&space;\sum_{i=0}^{N-1}&space;\frac{1}{2}s(i)^TQs(i)&space;&plus;&space;\frac{1}{2}q(i)^TRq(i)&space;&plus;&space;\frac{1}{2}s(N)^TPs(N)&space;\\&space;&&space;\text{subject&space;to}&space;&&space;&&space;s(0)&space;=&space;x_k,&space;\\&space;&&&&space;s(i&plus;1)&space;=&space;A&space;s(i)&space;&plus;&space;B&space;q(i),&space;\quad&space;\forall&space;i&space;\in&space;\{0,\ldots,N-1\},\\&space;&&&&space;u_\text{min}&space;\leq&space;q(i)&space;\leq&space;u_\text{max},&space;\quad\quad\quad\;\;&space;\forall&space;i&space;\in&space;\{0,\ldots,N-1\},\\&space;&&&&space;s(N)&space;\in&space;\mathbb{X}_f.&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;&&space;\underset{q,\,s}{\text{minimize}}&space;&&space;&&space;\sum_{i=0}^{N-1}&space;\frac{1}{2}s(i)^TQs(i)&space;&plus;&space;\frac{1}{2}q(i)^TRq(i)&space;&plus;&space;\frac{1}{2}s(N)^TPs(N)&space;\\&space;&&space;\text{subject&space;to}&space;&&space;&&space;s(0)&space;=&space;x_k,&space;\\&space;&&&&space;s(i&plus;1)&space;=&space;A&space;s(i)&space;&plus;&space;B&space;q(i),&space;\quad&space;\forall&space;i&space;\in&space;\{0,\ldots,N-1\},\\&space;&&&&space;u_\text{min}&space;\leq&space;q(i)&space;\leq&space;u_\text{max},&space;\quad\quad\quad\;\;&space;\forall&space;i&space;\in&space;\{0,\ldots,N-1\},\\&space;&&&&space;s(N)&space;\in&space;\mathbb{X}_f.&space;\end{aligned}" title="\begin{aligned} & \underset{q,\,s}{\text{minimize}} & & \sum_{i=0}^{N-1} \frac{1}{2}s(i)^TQs(i) + \frac{1}{2}q(i)^TRq(i) + \frac{1}{2}s(N)^TPs(N) \\ & \text{subject to} & & s(0) = x_k, \\ &&& s(i+1) = A s(i) + B q(i), \quad \forall i \in \{0,\ldots,N-1\},\\ &&& u_\text{min} \leq q(i) \leq u_\text{max}, \quad\quad\quad\;\; \forall i \in \{0,\ldots,N-1\},\\ &&& s(N) \in \mathbb{X}_f. \end{aligned}" /></a>
</center>

An rtPGM controller is created in Python based on the matrices defining the discrete-time linear system, its input bounds and the matrices defining the quadratic control objective:

```python
from rtPGM.controller import rtPGM
import numpy as np
# define system
A = np.array([[1.]])
B = np.array([[Ts]])
C = np.array([[1]])
D = np.array([[0]])
# define input bounds
umin = -2
umax = 2
# define objective
Q = 1e4*np.eye(1)
R = np.eye(1)
# horizon length
N = 20
# create rtPGM controller
rtpgm = rtPGM(A, B, C, D, Q, R, umin, umax, N)
```
The closed-loop behavior of the created controller is easily simulated in Python:

```python
x = np.array([[0]]) # initial state
r = 0.2 # output reference
for k in range(100):
    # evaluate rtPGM controller
    u = rtpgm.update(x, r)
    # simulate system
    x = A.dot(x) + B.dot(u)
```
In order to use your rtPGM controller on an embedded device, you can generate C++ code implementing your controller:

```python
rtpgm.codegen('rtPGM.h')
```
The C++ code is packed in a single header-file and can easily be integrated in your (embedded) C++ project:

```cpp
#include "rtPGM.h"
int main() {
    // create rtPGM controller
    rtPGM rtpgm;
    // define state, input and reference
    float x[1];
    float u[1];
    float r[1];
    r[0] = 0.12;
    while(true) {
        // estimate state ...
        x[0] = ... ;
        // evaluate rtPGM controller
        bool feasible = rtpgm.update(x, r, u);
        //apply u to the system ...
    }
    return 0;
}
```
## Experimental validation
The real-time PGM is implemented on an Arduino micro-controller to control the pendulum of the [mecotron](https://github.com/meco-group/mecotron). Click on a picture to watch the Youtube video.

<table style="border: none; border-collapse: collapse;" border="0" cellspacing="0" cellpadding="0" width="100%" align="center">
<tr>
<td align="center" valign="center" style="background-color:rgba(0, 0, 0, 0);">
<a href="https://www.youtube.com/watch?v=-XRa8bHNVRI">
<img src="https://img.youtube.com/vi/-XRa8bHNVRI/0.jpg" alt="Reference tracking pendulum"/>
</a>
</td>
<td align="center" valign="center" bgcolor="#FFFFFF">
<a href="https://www.youtube.com/watch?v=9mA5GPTmSVM">
<img src="https://img.youtube.com/vi/9mA5GPTmSVM/0.jpg" alt="Inverted pendulum control"/>
</a>
</td>
</tr>
</table>

## Installation
rtPGM is written in Python 2.7 and depends on the packages `numpy`, `scipy` and `matplotlib`:

```
sudo apt install python-numpy python-scipy python-matplotlib
```

rtPGM itself is installed by cloning the git repository and running the installation script:

```
git clone https://github.com/meco-group/rtPGM.git
cd rtPGM
sudo python setup.py --install
```

## Publications
Details on the implementation of the real-time PGM and a proof of closed-loop stability are found in the following publications:

* Van Parys R., Pipeleers G., *Real-time proximal gradient method for linear MPC*, European Control Conference, Limassol, Cyprus, 13-15 June 2018. [[pdf](https://lirias.kuleuven.be/bitstream/123456789/615194/1/ecc2018.pdf)]
