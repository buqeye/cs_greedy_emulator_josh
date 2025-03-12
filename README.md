# Greedy Emulators for Nuclear Two-Body Scattering

This is a repository for Joshua Maldonado's code. <br> This is based on the code used in his thesis, with adjustments from the paper with Dr. Christian Drischler, Dr. Dick Furnstahl, and Dr. Petár Mlinaríc.

This code accompanies the paper _Greedy Emulators for Nuclear Two-Body Scattering_.

## Code Breakdown

<img align="right" width="370" src="/markdown_figures/greedy-vs-POD.jpg">

There are three main files (classes) in this repository, the nuclear potential (Minnesota and GT+ Chiral) `/modules/Potential.py`, the Matrix Numerov method full order model `/modules/FOM.py`, and the Galerkin Projection based reduced order models `modules/ROM.py`.

This code is meant to be adaptable, _particularly_ `Potential.py`. Notably, there is a pre-built `DoItYourselfPotential` class that attempts to walk through how one would go about implementing their own _affine_ nuclear potential to be used with the solver and emulators.

The solvers laid out in `FOM.py` are strictly for the matrix Numerov method (with or without $a$ and $b$). Other solvers can be implemented to be used with the emulators, so long as the naming scheme of the newly implemented solver is consistent for the offline projections.

The emulators implemented in `ROM.py` are the Galerkin reduced order model (G-ROM) and the least-squares Petrov Galerkin reduced order model (LSPG-ROM). Both emulators have error estimators. In general, from my testing, without error estimation the G-ROM is faster than the LSPG-ROM, and with error estimation the LSPG-ROM is faster than the G-ROM. Although emulator speedups and runtimes vary with implementation and machines, this code has speedups (as run on _my_ machines) as follows:

|     CPU type     | FOM <br>(matrix Numerov) | G-ROM <br>| G-ROM w/ <br> error est. | LSPG-ROM <br>| LSPG-ROM w/ <br> error est. |
|:----------------:|:------------------------:|:---------:|:------------------------:|:------------:|:---------------------------:|
|        ARM       |              1           |    ~10x   |            ~4x           |     ~4.5x    |             ~4.5x           |
|        x86       |              1           |    ~7x    |           ~2.5x          |      ~3x     |              ~3x            |
| github codespace |              1           |   ~5.5x   |            ~2x           |     ~2.5x    |             ~2.5x           |


---
## Example Usage
```python
import numpy as np
import sys
sys.path.append("./modules")
from modules import Potential, FOM, ROM

# define the coordinate-space mesh
r = np.linspace(0, 12, 1000)

# initialize the potential (in this case the minnesota potential)
potential = Potential.Potential("minnesota", r, l=0)

# initialize the solver
solver = FOM.MatrixNumerovSolver(potential)

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

Creating the virtual environment (currently `numba` only wants to work with versions of python less than 3.13),

``` shell
python3.12 -m venv ./venv
```

using it,

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

This will test the computation of the potential, full order model, and matching process.


## Compiling the GT+ Chiral Potential

To compile the local chiral interactions GT+, first make sure you're in the [correct directory](https://github.com/Ub3rJosh/greedy-emulator/tree/main/chiral_construction),

``` shell
cd <your-path>/greedy-emulator/chiral_construction/
```

and using the GNU compiler (in this case, version 14),

``` shell
make clean
make CXX=g++-14
```

***NOTE: The compilation will _likely_ require that `ext_modules` be changed in `/chiral_construction/setup.py` or `chiral_construction/Makefile`***.

<!--
For compiling things that give the error: `m2 (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))` (at least for the purposes of looking at the [BUQEYE eigenvector continuation repo](https://github.com/buqeye/eigenvector-continuation) use these commands: \* `export LDFLAGS="-framework Accelerate"` \* `export NPY_DISTUTILS_APPEND_FLAGS=1` And then compile as mentioned in the BUQEYE repository but without the `-lliblapack` linker flag. This works only on MacOS computers.
-->

## Citing this work
```bibtex

```
