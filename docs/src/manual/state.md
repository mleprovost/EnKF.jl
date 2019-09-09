# Fields

```@meta
DocTestSetup = quote
  using ViscousFlow
  using Random
  Random.seed!(1)
end
```

```math
\def\ddt#1{\frac{\mathrm{d}#1}{\mathrm{d}t}}

\renewcommand{\vec}{\boldsymbol}
\newcommand{\uvec}[1]{\vec{\hat{#1}}}
\newcommand{\utangent}{\uvec{\tau}}
\newcommand{\unormal}{\uvec{n}}

\renewcommand{\d}{\,\mathrm{d}}
```


```@setup create
using Pkg
Pkg.activate("/media/mat/HDD/EnKF/")
using EnKF
using Plots
```
In `EnKF`, ensemble of members are  stored as a mutable structure `EnsembleState{N, TS}` where N is the number of members and `TS` an arbitrary type for each member. 
 
```@docs
EnsembleState
```


## Setting up your ensemble state

Let's see an example of creating a blank ensemble of 5 members of vector of length 10 and fill each member with a vector full of ones:

```@repl create
ens = EnsembleState(5, zeros(10))
fill!(ens, ones(10))
ens
```

`EnsembleState` supports three different constructors:

- The canonical one induced by the definition of `EnsembleState{N, TS}`
 by providing an `Array{TS,1}`:

```@repl create
ens = EnsembleState{5,Array{Float64,1}([rand(10) for i=1:5])
```

- Using `EnsembleState(N::Int, u)`, this one create an ensemble of `N` blank
members where each member is set to `zero(u)`:


```@repl create
ens = EnsembleState(5, ones(10))
```

- Using `EnsembleState(States::Array{TS,1})`, this one create an ensemble
of length `length(States)` and fill the i-th member with `States[i]`:

Any type of state is supported by `EnKF` even your own data structures!

```@repl create
struct MyArray
	B::Array{ComplexF64,1}
end
```




## Operations on EnsembleState

## Methods

```@autodocs
Modules = [Fields]
Order   = [:type, :function]
```

## Index

```@index
Pages = ["states.md"]
```
