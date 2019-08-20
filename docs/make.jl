using Pkg
#include("/media/mat/HDD/EnKF/src/EnKF.jl")
Pkg.activate("/media/mat/HDD/EnKF/")
using Documenter, EnKF, ViscousFlow
makedocs(
    sitename = "EnKF.jl",
    pages = [
        "Home" => "index.md",
        "Manual" => ["manual/inflation.md",
                     "manual/state.md",
                     "manual/system.md",
                     "manual/timemarching.md"
                     ],
#        "Internals" => [ "internals/properties.md"]
    ],
   format = Documenter.HTML(assets = ["assets/custom.css"]),
#   strict = true
)

if "DOCUMENTER_KEY" in keys(ENV)
    deploydocs(;
     repo = "github.com/mleprovost/EnKF.jl.git",
    )
end
