using PRISM
using LinearAlgebra

@testset "Volume object" begin
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
    @test all(PRISM.real2voxel(v, p) .≈ p)
end

@testset "Ray intersect" begin
    nx, ny, nz = 4, 5, 4
    edges = [0:nx, 0:ny, 0:nz]
    v = Volume(edges)

    x = y = z = 1.5:1:5.5
    pa = PointArray(x,y,z)
    pa2 = PRISM.real2voxel(v, pa)
    @test all([p.x for p in pa2] .== x)

    # With a random pattern, check that z-aligned paths agree
    v.densities[:,:,:] = rand(nx, ny, nz)
    @test PRISM.path_integrated_density(v, [1.5, 1.5, 0], [1.5, 1.5, nz]) ≈ sum(v.densities[2, 2, :])
    @test PRISM.path_integrated_density(v, [1.5, 3.5, 0], [1.5, 3.5, nz]) ≈ sum(v.densities[2, 4, :])
    @test PRISM.path_integrated_density_fast(v, [1.5, 1.5, 0], [1.5, 1.5, nz]) ≈ sum(v.densities[2, 2, :])
    @test PRISM.path_integrated_density_fast(v, [1.5, 3.5, 0], [1.5, 3.5, nz]) ≈ sum(v.densities[2, 4, :])

    # Now check a few random positions where the line goes from one y-plane pixel edge to the next
    Ntests = 10
    for (x,y) in zip(nx*rand(Ntests), rand(1:ny, Ntests))
        p3 = [x, y-1, 0]
        p4 = [x, y, nz]
        expected = sum(v.densities[1+Int(floor(x)), y, :]) * norm(p4-p3)/nz
        @test PRISM.path_integrated_density(v, p3, p4) ≈ expected
        @test PRISM.path_integrated_density_fast(v, p3, p4) ≈ expected
    end

    pA = [2, 3, 0]
    pB = [3, 4, nz]
    expected = sum(v.densities[3, 4, :]) * norm(pB-pA)/nz
    @test PRISM.path_integrated_density(v, pA, pB) ≈ expected
    @test PRISM.path_integrated_density_fast(v, pA, pB) ≈ expected
    pC = [4, 5, nz]
    dsum = sum(v.densities[3, 4, 1:div(nz, 2)]) + sum(v.densities[4, 5, (div(nz, 2)+1):end])
    expected = dsum * norm(pC-pA)/nz
    @test PRISM.path_integrated_density(v, pA, pC) ≈ expected
    @test PRISM.path_integrated_density_fast(v, pA, pC) ≈ expected

    # Repeat, but using the bulk (PointArray) interface.
    Ntests = 100
    x = nx*rand(Ntests)
    y = rand(1:ny, Ntests)
    pa3 = PointArray(x, y.-1, zeros(Ntests))
    pa4 = PointArray(x, y, nz.+zeros(Ntests))
    n = norm(pa4-pa3)
    expected_integral = [sum(v.densities[1+Int(floor(x[i])), y[i], :]) * n[i] /nz for i=1:Ntests]
    @test all(PRISM.path_integrated_density(v, pa3, pa4) .≈ expected_integral)
    @test all(PRISM.path_integrated_density_fast(v, pa3, pa4) .≈ expected_integral)

    # Do some tests that cross at known angles (45°)
    x_on_y_axis = -ny-1:nx+1
    for sx in x_on_y_axis
        s = [sx-1, -1, 0.5]
        e = [sx+ny+1, ny+1, 0.5]
        S, E = PRISM.enter_exit_points(v, s, e)
        expect = 0.0
        if sx > -ny && sx < nx
            if sx ≤ 0
                expect = sqrt(2)*sum([v.densities[i,i-sx,1] for i=1:min(ny+sx, nx)])
            else    
                expect = sqrt(2)*sum([v.densities[i+sx,i,1] for i=1:min(ny, nx-sx)])
            end
        end
        @test isapprox(PRISM.path_integrated_density(v, s, e), expect; atol=1e-30, rtol=√eps(Float64))
        @test isapprox(PRISM.path_integrated_density_fast(v, s, e), expect; atol=1e-30, rtol=√eps(Float64))
    end

    s = [0, 0.5, 2.5]
    e = [nx, nx+0.5, 2.5]
    expect = sqrt(0.5)*sum([v.densities[i,i,3] + v.densities[i,i+1,3] for i=1:min(nx, ny)])
    @test PRISM.path_integrated_density(v, s, e) ≈ expect
    @test PRISM.path_integrated_density_fast(v, s, e) ≈ expect

    s = [-.5, 0, 2.5]
    e = [nx, nx+0.5, 2.5]
    expect = sqrt(0.5)*sum([v.densities[i,i,3] + v.densities[i,i+1,3] for i=1:min(nx, ny)])
    @test PRISM.path_integrated_density(v, s, e) ≈ expect
    @test PRISM.path_integrated_density_fast(v, s, e) ≈ expect
    
    zplane_norm = norm(e-s)
    s = [-.5, 0, 2.0]
    e = [nx, nx+0.5, 3.0]
    expect = sqrt(0.5)*sum([v.densities[i,i,3] + v.densities[i,i+1,3] for i=1:min(nx, ny)])
    expect *= norm(s-e)/zplane_norm
    d = PRISM.path_integrated_density(v, s, e)
    df = PRISM.path_integrated_density_fast(v, s, e)
    @test d ≈ expect
    @test df ≈ expect

    # Test rays heading in the -y direction through 1, 2, or 3 pixels
    for xstart in [.1, .5, .9]
        yend = .2:.2:3
        e = PointArray(0.0, yend, 0.5)
        s = PointArray(xstart, yend*0, 0.5)
        expect = [norm(e1-s1)*v.densities[1] for (s1,e1) in zip(s,e)]
        for (i,y) in enumerate(yend)
            y ≤ 1 && continue
            mid = xstart*(1 - 1 / y)
            expect[i] = norm(s[i]-[mid,1,.5])  * v.densities[1]
            if y ≤ 2
                expect[i] += norm(e[i]-[mid,1,.5])  * v.densities[1,2,1]
            else
                mid2 = xstart*(1 - 2 / y)
                expect[i] += norm([mid2,2,.5]-[mid,1,.5])  * v.densities[1,2,1]
                expect[i] += norm(e[i]-[mid2,2,.5])  * v.densities[1,3,1]
            end
        end
        d = PRISM.path_integrated_density(v, s, e)
        df = PRISM.path_integrated_density_fast(v, s, e)
        @test all(d .≈ expect)
        @test all(df .≈ expect)
    end
end

@testset "merge_sorted" begin
    # Try with two ranges and one true array.
    a = 1:3:10
    b = 2:3:10
    c = [3, 6, 9]
    merge_sorted = PRISM.merge_sorted
    @test all(merge_sorted(a,b) .== [1,2,4,5,7,8,10])
    @test all(merge_sorted(b,a) .== [1,2,4,5,7,8,10])
    @test all(merge_sorted(a,c) .== [1,3,4,6,7,9,10])
    @test all(merge_sorted(a,b,c) .== [1,2,3,4,5,6,7,8,9,10])
    @test all(merge_sorted(a,b,c,c,a,b) .== [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10])
    @test all(merge_sorted(b, []) .== b)
end