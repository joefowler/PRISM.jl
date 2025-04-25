using PRISM
using LinearAlgebra

@testset "PointArray" begin
    pf = PointArray([1,2,3], [1,2,3,], [1,2, 5.9])
    @test eltype(pf[1]) == Float64
    @test_throws DimensionMismatch PointArray([1,2,3], [1,2,3,4], [1,2])

    # Test construction mixing real numbers and vectors
    npt = 5
    a1 = 1:npt
    a2 = [-3, 2, 10, 4.4, 14]
    tests = [
        [8, 7, 5, 1],
        [a1, 0, 5, npt],
        [3, a1, 4, npt],
        [0, 5, a1, npt],
        [a1, a2, 5, npt],
        [a2, a1, 4, npt],
        [0, a2, a1, npt],
        [a2, 5, a1, npt],
        [a1, a2, a1, npt],
    ]
    for (x, y, z, L) in tests
        p = PointArray(x, y, z)
        @test all([a.x for a in p] .== x)
        @test all([a.y for a in p] .== y)
        @test all([a.z for a in p] .== z)
        @test length(p) == L
    end
end