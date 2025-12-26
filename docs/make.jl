using Documenter
using Momenta

makedocs(
    sitename = "Momenta.jl",
    format = Documenter.HTML(),
    modules = [Momenta],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/dazhwu/Momenta.jl.git",
)