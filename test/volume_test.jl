using PRISM

@testset "volume_object" begin
    @testset "contructors" begin
        nx, ny, nz = 3, 5, 4
        edges = [0:nx, 0:ny, 0:nz]
        v1 = Volume(edges, zeros(Float64, nx, ny, nz), [nx, ny, nz], nx*ny*nz)
        v2 = Volume(edges)
        @test all(v2.densities .== 0.0)
        @test v1.voxelsTotal == v2.voxelsTotal
        @test all(v1.voxelsPerEdge .== v2.voxelsPerEdge)
    end
    nx, ny, nz = 3, 5, 4
    edges = [0:nx, 0:ny, 0:nz]
    v = Volume(edges)
    @test PRISM.corners(v) == [[0,nx],[0,ny], [0,nz]]
    p = [1.5, 2.5, 1.3]
    @test all(PRISM.voxelScale(v, p) .≈ p)
end