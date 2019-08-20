## EnKF.jl

_A framework for data assimilation with ensemble Kalman filter_

| Documentation | Build Status |
|:---:|:---:|
| [![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://mleprovost.github.io/EnKF.jl/latest) | [![Build Status](https://img.shields.io/travis/mleprovost/EnKF.jl/master.svg?label=linux)](https://travis-ci.org/mleprovost/EnKF.jl) [![Build status](https://img.shields.io/appveyor/ci/jdeldre/whirl-jl/master.svg?label=windows)](https://ci.appveyor.com/project/mleprovost/EnKF/branch/master) [![codecov](https://codecov.io/gh/mleprovost/EnKF.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mleprovost/EnKF.jl) |

## About the package

The purpose of this package is to enable easy setup and solution of viscous incompressible flows. Documentation can be found at https://jdeldre.github.io/ViscousFlow.jl/latest.

**ViscousFlow.jl** is registered in the general Julia registry. To install in julia `0.6`, type
```julia
julia> Pkg.add("ViscousFlow")
```
in the Julia REPL.

In julia `0.7` or `1.0`, enter the package manager by typing `]` and then type,
e.g.,
```julia
(v1.0) pkg> add EnKF
```

Then, in any version, type
```julia
julia> using EnKF
```
For examples, consult the documentation or see the example Jupyter notebooks in the Examples folder.

![](https://github.com/jdeldre/ViscousFlow.jl/raw/master/cylinderRe400.gif)
