# EnKF.jl

_A framework for data assimilation using ensemble Kalman filter in Julia_

The objective of this package is to allow a easy and fast setup of data
assimilation problems using the ensemble Kalman filter. This package provides
tools for: 

- constructing data structure for an ensemble of members
- applying covariance inflation (additive, multiplicative, multiplico-additive...) on these ensemble
- setting the data assimilation problem for linear/nonlinear system with linear/nonlinear measurements


## Installation

This package works on Julia `1.0` and above and is registered in the general Julia registry. 

In julia `1.0`, enter the package manager by typing `]` and then type,
e.g.,
```julia
(v1.0) pkg> add EnKF
```

Then, in any version, type
```julia
julia> using EnKF
```

The plots in this documentation are generated using [Plots.jl](http://docs.juliaplots.org/latest/).
You might want to install that, too, to follow the examples.

## References

[^1]: Evensen, Geir. "The ensemble Kalman filter: Theoretical formulation and practical implementation." Ocean dynamics 53.4 (2003): 343-367.

[^2]: Asch, Mark, Marc Bocquet, and Maëlle Nodet. Data assimilation: methods, algorithms, and applications. Vol. 11. SIAM, 2016.

