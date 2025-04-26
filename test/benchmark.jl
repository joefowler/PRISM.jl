using PRISM
using BenchmarkTools

"""
Usage from the PRISM.jl main directory. Assumes you have already installed BenchmarkTools in your main
environment. (It is not required by this package)

    julia test/benchmark.jl
"""

function create_volume()
    V = Volume([-5:.04:5, -5:.04:5, 8:.08:12])
    V.densities[:,:,:] = rand(V.voxelsPerEdge...)
    for i=30:40:250
        for j=30:40:250
            V.densities[i-20:i, j-20:j, 20:45] .= 1
        end
    end
    V
end

function benchmark_integrate(npt::Integer)
    R = create_volume()
    x = randn(npt)*50
    y = randn(npt)*50
    z = fill(250.0, npt)
    P = PointArray(x, y, z)
    origin = 0*P
    R, origin, P
end

if abspath(PROGRAM_FILE) == @__FILE__
    R, origin, P = benchmark_integrate(100000)
    @btime d = PRISM.path_integrated_density(R, origin, P)
    @btime d = PRISM.path_integrated_density_fast(R, origin, P)
end
