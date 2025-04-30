# Greedy Emulators for Nuclear Two-Body Scattering

This is a repository for Joshua Maldonado's code. <br> This is based on the code used in [his thesis](https://etd.ohiolink.edu/acprod/odb_etd/r/etd/search/10?p10_accession_num=ohiou1726590160450187&clear=10&session=103007851808756), with adjustments from the paper with [_Greedy Emulators for Nuclear Two-Body Scattering_](https://arxiv.org/abs/2504.06092) by Joshua Maldonado, Dr. Christian Drischler, Dr. Dick Furnstahl, and Dr. Petár Mlinaríc.

## Code Breakdown

<img align="right" width="375" src="/markdown_figures/greedy-vs-POD.jpg">

There are three main files (classes) in this repository, the nuclear potentials Minnesota and GT+ local chiral (as laid out in [Gezerlis et. al.](https://doi.org/10.1103/PhysRevC.90.054323)) `/modules/Potential.py`, the Matrix Numerov method full order model `/modules/FOM.py`, and the Galerkin Projection based reduced order models `/modules/ROM.py`.

This code is meant to be adaptable, _particularly_ `Potential.py`. Notably, there is a pre-built `DoItYourselfPotential` class that attempts to walk through how one would go about implementing their own _affine_ nuclear potential to be used with the solver and emulators.

The solver laid out in `FOM.py` are strictly for the matrix Numerov method (with or without $a$ and $b$, as well as the all-at-once method). Other solvers can be implemented to be used with the emulators, such as the ones laid out in the appendix, so long as the naming scheme of the newly implemented solver is consistent for the offline projections.

The emulators implemented in `ROM.py` are the Galerkin reduced order model (G-ROM) and the least-squares Petrov-Galerkin reduced order model (LSPG-ROM). Both emulators have error estimators. In general, from my testing, without error estimation the G-ROM is faster than the LSPG-ROM, and with error estimation the LSPG-ROM is faster than the G-ROM. Although emulator speedups and runtimes vary with implementation and machines, this code has speedups (as run on _my_ machines) as follows:

|     CPU type     | FOM <br>(matrix Numerov) | G-ROM <br>| G-ROM w/ <br> error est. | LSPG-ROM <br>| LSPG-ROM w/ <br> error est. |
|:----------------:|:------------------------:|:---------:|:------------------------:|:------------:|:---------------------------:|
|        ARM       |          1 (~90µs)       |    ~9x    |            ~6x           |     ~4.5x    |             ~4.5x           | <!-- M2 Max -->
|        x86       |          1 (~155µs)      |    ~7x    |            ~4x           |      ~3x     |              ~3x            | <!-- Intel i5 4690K (overclocked)-->
| github codespace |          1 (~145µs)      |    ~6x    |           ~4.5x          |     ~2.5x    |             ~2.5x           |

The reported "~__x" values are the approximate speedups _this_ implementation of the emulators and matrix Numerov FOM. The values noted in parenthesis under the FOM header are the approximate runtimes of the matrix Numerov method on each system. These speedups were taken by looking at 5-10 snapshots when using the Minnesota potential and the GT+ local chiral potential; each case was run for $7 \times 50,000$ solutions using `%timeit` in the Jupyter notebook `runtimes.ipynb`. You can run this notebook to see the runtimes and speedups on your system!

---
## Example Usage
```python
import numpy as np
from modules import Potential, FOM, ROM

# define the coordinate-space mesh
r = np.linspace(0, 12, 1000)

# initialize the potential (in this case the minnesota potential)
potential = Potential.Potential("minnesota", r, l=0)

# initialize the solver
energy = 50  # center-of-mass energy in MeV
solver = FOM.MatrixNumerovSolver(potential, energy=energy)

# initialize the emulator 
parameter_bounds = {"V_s": [-400., 0.]}  # Vary the parameter "V_s" between -400 MeV and 0 MeV
emulator = ROM.Emulator(parameter_bounds, solver)

# construct emulator basis and perform offline projections
emulator.train()

# visualize accuracy of basis construction with convergence plot
emulator.convergence_plot()

# visualize wave functions and error across coordinate-space
emulator.emulation_errors_at_theta(potential.default_theta)
```



---

## Python Environment

Creating the virtual environment,

``` shell
python3 -m venv ./venv
```

activating it,

```shell
source ./venv/bin/activate
```

and installing the required packages,

```shell
pip3 install -r requirements.txt
```

## Testing 
To test the installation and compilation of the repository, including the compilation of the local GT+ chiral potential, one can run
```shell
python3 unit_test.py
```

This will test the computation of the potential, full order model, and matching process and ensure that the expected results are obtained.


## Compiling the GT+ Chiral Potential

To compile the local chiral interactions GT+, first make sure you're in the [correct directory](https://github.com/buqeye/cs_greedy_emulator_josh/tree/main/chiral_construction),

``` shell
cd <your-path>/greedy-emulator/chiral_construction/
```

and using the GNU compiler (in this case, version 14),

``` shell
make clean
make CXX=g++-14
```

***NOTE: The compilation will _likely_ require that `ext_modules` be changed in `/chiral_construction/setup.py` or `chiral_construction/Makefile`***.

Further details regarding the compilation of this potential can be found at https://github.com/cdrischler/general_kvp.

<!--
For compiling things that give the error: `m2 (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))` (at least for the purposes of looking at the [BUQEYE eigenvector continuation repo](https://github.com/buqeye/eigenvector-continuation) use these commands: \* `export LDFLAGS="-framework Accelerate"` \* `export NPY_DISTUTILS_APPEND_FLAGS=1` And then compile as mentioned in the BUQEYE repository but without the `-lliblapack` linker flag. This works only on MacOS computers.
-->

## Details on the Kohn Anomaly Figure
The end of the results section has a figure (Fig. 11) that shows the detection and elimination of an anomaly by the G-ROM, and simultaneously shows that the LSPG-ROM does not see such an anomaly. 
This figure was created using the $^1S_0$ GT+ Chiral Potential with cutoffs of $r=1\rm{fm}, \Lambda=1000\rm{MeV}$. 
A one-dimensional parameter space is used for the making of this figure, varying $C_S$ between $150$% of its best fit value of 5.4385 fm $^2$. 
For each emulator, two snapshots were added on the boundary of this parameter space. 
The emulators then added one snapshot to their basis using the greedy algorithm laid out in this work. 
The creation of this figure (with notes and annotations) are included in `kohn_anomaly_visualization.ipynb`.


## Citing this work
```bibtex
@article{Maldonado:2025ftg,
    author = "Maldonado, J. M. and Drischler, C. and Furnstahl, R. J. and Mlinari\'c, P.",
    title = "{Greedy Emulators for Nuclear Two-Body Scattering}",
    eprint = "2504.06092",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    month = "4",
    year = "2025"
}
```
