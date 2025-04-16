using PRISM
using LinearAlgebra

@testset "PointArray" begin
    @test_throws AssertionError PointArray([1,2,3], [1,2,3,4], [1,2])
    x = 1:4
    y = x.+2
    z = x.+1
    pa = PointArray(x,y,z)
    @test all(pa.x .== x)
    @test all(pa.y .== y)
    @test all(pa.z .== z)
    d = pa - pa
    @test all(d.x .== 0)
    @test all(d.y .== 0)
    @test all(d.z .== 0)
    s = pa + pa
    @test all(s.x .== 2x)
    @test all(s.y .== 2y)
    @test all(s.z .== 2z)
    for double in [2pa, pa*2]
        @test all(double.x .== 2x)
        @test all(double.y .== 2y)
        @test all(double.z .== 2z)
    end
end

@testset "volume_object" begin
    nx, ny, nz = 4, 5, 4
    edges = [0:nx, 0:ny, 0:nz]
    v = Volume(edges)
    
    @testset "contructors" begin
        v1 = Volume(edges, zeros(Float64, nx, ny, nz), [nx, ny, nz], nx*ny*nz)
        @test all(v.densities .== 0.0)
        @test v1.voxelsTotal == v.voxelsTotal
        @test all(v1.voxelsPerEdge .== v.voxelsPerEdge)
    end

    @test PRISM.corners(v) == [[0,nx],[0,ny], [0,nz]]
    p = [1.5, 2.5, 1.3]
    @test all(PRISM.voxelScale(v, p) .≈ p)

    x = y = z = 1:3
    pa = PointArray(x,y,z)
    pa2 = PRISM.voxelScale(v, pa)
    @test all(pa2.x .== x)

    # With a random pattern, check that z-aligned paths agree
    v.densities[:,:,:] = rand(nx, ny, nz)
    @test PRISM.path_integrated_density(v, [1.5, 1.5, 0], [1.5, 1.5, nz]) ≈ sum(v.densities[2, 2, :])
    @test PRISM.path_integrated_density(v, [1.5, 3.5, 0], [1.5, 3.5, nz]) ≈ sum(v.densities[2, 4, :])

    # Now check a few random positions where the line goes from one y-plane pixel edge to the next
    Ntests = 5
    for (x,y) in zip(nx*rand(Ntests), rand(1:ny, Ntests))
        p3 = [x, y-1, 0]
        p4 = [x, y, nz]
        @test PRISM.path_integrated_density(v, p3, p4) ≈ sum(v.densities[1+Int(floor(x)), y, :]) * norm(p4-p3)/nz
    end

    pA = [2, 3, 0]
    pB = [3, 4, nz]
    @test PRISM.path_integrated_density(v, pA, pB) ≈ sum(v.densities[3, 4, :]) * norm(pB-pA)/nz
    pC = [4, 5, nz]
    dsum = sum(v.densities[3, 4, 1:div(nz, 2)]) + sum(v.densities[4, 5, (div(nz, 2)+1):end])
    @test PRISM.path_integrated_density(v, pA, pC) ≈ dsum * norm(pC-pA)/nz
end