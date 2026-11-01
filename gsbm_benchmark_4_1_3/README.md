# Steady Poisson: moving boundaries (4.1.3)

The steady Poisson problem of [4.1.1](../gsbm_benchmark_4_1_1/README.md), with its manufactured solution, on two circles of radius $R = 0.15$ [1]: one fixed around $(0.2, 0.5)$, the other around $(0.5 + d, 0.5)$, for $d$ from $-0.2$ to $0.2$ in steps of $0.0125$. The circles are apart for $d > 0$, touch at $(0.35, 0.5)$ for $d = 0$, and merge for $d < 0$: the topology changes at $d = 0$. Every $d$ is a steady problem of its own, on the same background of $100 \times 100$ knot spans.

`skins.py` writes the skin of every $d$ to `config/skins/`, as exact NURBS circles: for $d \geq 0$ the two circles, and for $d < 0$ the boundary of their union, the arc of each circle outside the other, meeting at the two points where they intersect.

The method is that of 4.1.1: the skew-symmetric Nitsche method without penalty, the enhanced shift of order $2p$ and the ghost penalty $\kappa / \big((p!)^2 (2p+1)\big)$ with $\kappa = 0.5$, in `config/MaterialParameters_skew_p<p>.json`.

## Error along the sweep

From this folder, with Kratos on the Python path (writes `out/sweep_p<p>.csv` and the fields of the animation):

```bash
OMP_NUM_THREADS=1 python3 sweep.py config/ProjectParameters_circles_skew.json --degrees 1 2 3 4 5 --materials "config/MaterialParameters_skew_p{p}.json"
python3 MainKratos.py config/ProjectParameters_circles_skew.json --degrees 5 --materials "config/MaterialParameters_skew_p{p}.json"
python3 figures.py
```

![](out/gsbm_circles_animation.gif)

At no degree does the error change with the topology at $d = 0$, nor between the separate ($d > 0$) and merged ($d < 0$) circles. Its medians over $d$ are $9.2 \cdot 10^{-2}$, $1.1 \cdot 10^{-2}$, $1.1 \cdot 10^{-3}$, $2.8 \cdot 10^{-4}$ and $2.3 \cdot 10^{-5}$ for $p = 1$ to $5$. As $d$ advances in steps of $1.25\,h$, the leftmost and rightmost points of the moving circle alternate between knot lines and a quarter of a knot span off them. At $p = 5$ the error alternates with them, between about $10^{-5}$ and $10^{-3}$: these are the jumps from one mesh to the next of 4.1.1. The two peaks, $p = 4$ at $d = -0.05$ and $p = 5$ at $d = -0.0375$, are set by round-off. With 4 OpenMP threads instead of 1 they fall from $5.2 \cdot 10^{-3}$ to $3.1 \cdot 10^{-4}$ and from $3.2 \cdot 10^{-2}$ to $2.1 \cdot 10^{-3}$, while at $p = 5$, $d = -0.025$, beside them, the error is the same to two digits. The figure comes from single-thread runs, `OMP_NUM_THREADS=1`, which reproduce it exactly.

At $p = 1$ the error is a median 1.9 to 3.9 times that of the paper's weights below, which come from the symmetric Nitsche method with its penalty.

The $p = 1$ points are the results of [1], computed with the authors' Julia repository, [TheGeneralizedSBM.jl](https://github.com/CMOE-TUDelft/TheGeneralizedSBM.jl) [2], for the WSBM ($\alpha_h$), SBM ($\tilde{\alpha}_h$) and O-SBM ($\hat{\alpha}_h$, $\lambda = 0.5$) weights. Their absolute $L^2$ errors (Eq. 48) are in `config/reference_errors_p1.csv`, divided by $\Vert u \Vert$ in the figure.

## Ghost regions at $p$-degree

The fields at every degree on the $100 \times 100$ background, written to `out/circles_skew_p<p>_d<d>.vtk`:

```bash
python3 MainKratos.py config/ProjectParameters_circles_skew.json --degrees 1 2 3 4 --materials "config/MaterialParameters_skew_p{p}.json"
python3 figures.py
```

![](out/gsbm_circles_degrees.gif)

## The shifted subscale

As in [4.1.1](../gsbm_benchmark_4_1_1/README.md#the-shifted-subscale): the symmetric Nitsche method of [1], with its penalty $\chi (p+1)^2$, $\chi = 20$, the enhanced shift and the ghost penalty, with a **shifted subscale**: the shift rewritten with the equation, expanded from the closest point on the true boundary. It is computed at the same positions on the same background.

![](out/gsbm_circles_subscale.png)

Its errors are a median 0.55, 0.67, 0.45, 0.28 and 0.36 times those of the skew-symmetric method for $p = 1$ to $5$, and they hardly change along the sweep. At $p = 5$ they stay between $6.9 \cdot 10^{-6}$ and $1.5 \cdot 10^{-5}$, against $1.2 \cdot 10^{-5}$ to $1.4 \cdot 10^{-3}$ for the skew-symmetric method, its round-off peak at $d = -0.0375$ aside. The exceptions are the two positions where the circles touch or almost touch, $d = 0$ and $0.0125$, with $1.8 \cdot 10^{-5}$ and $4.2 \cdot 10^{-5}$. At $p = 1$ they are close to the paper's: a median 1.3, 1.1 and 2.0 times its WSBM, SBM and O-SBM errors.

## References

[1] O. Colomés, J. Modderman, G. Scovazzi. The generalized shifted boundary method for geometry-parametric PDEs and time-dependent domains. *Comput. Methods Appl. Mech. Engrg.* 452 (2026) 118748.

[2] O. Colomés, J. Modderman. TheGeneralizedSBM.jl: a Julia-based implementation of the Generalized Shifted Boundary Method using Gridap.jl (2025), https://github.com/CMOE-TUDelft/TheGeneralizedSBM.jl.
