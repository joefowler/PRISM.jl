using LinearAlgebra

struct PointArray
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    n::Int
end

function PointArray(x::AbstractVector, y::AbstractVector, z::AbstractVector)
    n = length(x)
    @assert n == length(y)
    @assert n == length(z)
    PointArray(x, y, z, n)
end
PointArray(x::Number, y::Number, z::Number) = PointArray([x], [y], [z])
PointArray(a::AbstractMatrix) = PointArray(a[:,1], a[:,2], a[:,3])
PointArray(v::AbstractVector) = PointArray(v...)
Base.getindex(p::PointArray, i::Integer) = [p.x[i], p.y[i], p.z[i]]
matrix(p::PointArray) = hcat(p.x, p.y, p.z)

Base.:+(a::PointArray, b::PointArray) = PointArray(a.x.+b.x, a.y.+b.y, a.z.+b.z, a.n)
Base.:-(a::PointArray, b::PointArray) = PointArray(a.x.-b.x, a.y.-b.y, a.z.-b.z, a.n)
Base.:*(k::Real, p::PointArray) = PointArray(k*p.x, k*p.y, k*p.z, p.n)
Base.:*(p::PointArray, k::Real) = Base.:*(k, p)

# These are equivalent to scaling each point in the PointArray by a different scalar
Base.:*(p::PointArray, v::AbstractVector) = PointArray(v.*p.x, v.*p.y, v.*p.z)
Base.:*(v::AbstractVector, p::PointArray) = p*v
LinearAlgebra.norm(p::PointArray) = sqrt.(p.x.^2 .+ p.y.^2 .+ p.z.^2)
LinearAlgebra.norm(p::PointArray, q::Real) = sum([abs.(a).^q for a in (p.x, p.y, p.z)]).^(1/q)

struct Volume
    edges::Vector
    densities::Array{Float64}
    voxelsPerEdge::Vector{Int}
    voxelsTotal::Int
end

function Volume(edges::AbstractVector)
    @assert length(edges) == 3
    voxelsPerEdge = [length(e)-1 for e in edges]
    densities = zeros(Float64, voxelsPerEdge...)
    Volume(edges, densities)
end

function Volume(edges::AbstractVector, densities::AbstractArray)
    @assert length(edges) == 3
    @assert length(size(densities)) == 3
    voxelsPerEdge = [length(e)-1 for e in edges]
    voxelsTotal = prod(voxelsPerEdge)
    Volume(edges, densities, voxelsPerEdge, voxelsTotal)
end

corners(v::Volume) = [[minimum(x), maximum(x)] for x in v.edges]

function voxelScale(v::Volume, pa::PointArray)
    ex, ey, ez = v.edges
    x = (pa.x .- minimum(ex)) / step(ex)
    y = (pa.y .- minimum(ey)) / step(ey)
    z = (pa.z .- minimum(ez)) / step(ez)
    PointArray(x, y, z, pa.n)
end
voxelScale(v::Volume, p::AbstractArray) = [(x-minimum(e))/step(e) for (x,e) in zip(p, v.edges)]

function enter_exit_points(v::Volume, p1::AbstractArray, p2::AbstractArray)
    @assert length(p1) == 3
    @assert length(p2) == 3
    @assert p1[3] != p2[3]

    dx = p2 - p1
    zmin, zmax = minimum(v.edges[3]), maximum(v.edges[3])
    a1 = (zmin - p1[3]) / dx[3]
    a2 = (zmax - p1[3]) / dx[3]
    p1 + a1*dx, p1 + a2*dx
end

function enter_exit_points(v::Volume, pa1::PointArray, pa2::PointArray)
    Δp = pa2 - pa1
    zmin, zmax = minimum(v.edges[3]), maximum(v.edges[3])
    a1 = (zmin .- pa1.z) ./ Δp.z
    a2 = (zmax .- pa1.z) ./ Δp.z
    pa1 + a1*Δp, pa1 + a2*Δp
end

function path_integrated_density(v::Volume, p1::AbstractArray, p2::AbstractArray)
    pfront, pback = enter_exit_points(v, p1, p2)
    C1, C2, _ = corners(v)
    involume1 = pfront[1] ≥ C1[1] && pfront[1] ≤ C1[2] && pfront[2] ≥ C2[1] && pfront[2] ≤ C2[2]
    involume2 = pback[1] ≥ C1[1] && pback[1] ≤ C1[2] && pback[2] ≥ C2[1] && pback[2] ≤ C2[2]
    !(involume1 || involume2) && return 0.0
    total_distance = norm(pback-pfront)
    
    # Let α=[0,1] represent [pfront,pback]. Now find all values of α where a pixel boundary is crossed
    # Start with all crossings of z-plane boundaries
    α = collect(LinRange(0, 1, 1+v.voxelsPerEdge[3]))

    vfront = voxelScale(v, pfront)
    vback = voxelScale(v, pback)
    dv = vback - vfront

    for axis in (1,2)
        iffront = Int(floor(vfront[axis]))
        ifback = Int(floor(vback[axis]))
        if iffront != ifback
            sign = +1
            if iffront > ifback
                sign = -1
            end
            crossings = iffront+sign:sign:ifback
            dvox = vback[axis]-vfront[axis]
            for c in crossings
                push!(α, (c - vfront[axis]) / dvox)
            end
        end
    end
    sort!(α)

    integral = 0.0
    if length(α) ≥ 2
        prevα = α[1]
        for nextα in α[2:end]
            thisdist = total_distance*abs(nextα-prevα)
            thisdist ≤ 0 && continue
            voxelID = [1+Int(floor(x)) for x in (vfront + dv*prevα)]
            any(voxelID .< 1) && continue
            any(voxelID .> v.voxelsPerEdge) && continue
            integral += thisdist * v.densities[voxelID...]
            prevα = nextα
        end
    end
    integral
end

    """merge_sorted(a::AbstractVector, b::AbstractVector, [c::AbstractVector])

    Assumes that both `a` and `b` are pre-sorted vectors of the same type.
    Returns an array that contains all elements of either input, sorted. If `c` is also given, merge all 3 arrays.
    """
function merge_sorted(a::AbstractVector, b::AbstractVector)
    na = length(a)
    nb = length(b)
    na == 0 && return b
    nb == 0 && return a
    T = promote_type(eltype(a), eltype(b))
    merged = Array{T}(undef, na+nb)
    i = ia = ib = 1
    while ia ≤ na && ib ≤ nb
        if a[ia] < b[ib]
            merged[i] = a[ia]
            ia += 1
        else
            merged[i] = b[ib]
            ib += 1
        end
        i += 1
    end
    
    if ia ≤ na
        merged[i:end] = a[ia:end]
    end
    if ib ≤ nb
        merged[i:end] = b[ib:end]
    end
    merged
end
merge_sorted(a::AbstractVector, b::AbstractVector, c::AbstractVector) = merge_sorted(a, merge_sorted(b, c))

function path_integrated_density(v::Volume, p1::PointArray, p2::PointArray)
    pfront, pback = enter_exit_points(v, p1, p2)
    integral = zeros(Float64, p1.n)
    total_distance = norm(pback-pfront)

    vfront = voxelScale(v, pfront)
    vback = voxelScale(v, pback)
    dv = vback - vfront
    intfloor(x::Real) = Int(floor(x))

    for i in 1:p1.n
        # Let α=[0,1] represent [pfront,pback]. Now find all values of α where a pixel boundary is crossed
        # Start with all crossings of z-plane boundaries
        αz = LinRange(0, 1, 1+v.voxelsPerEdge[3])
        αx = Float64[]
        αy = Float64[]

        # Find all crossings of x-plane boundaries
        fx = vfront.x[i]
        bx = vback.x[i]
        ifx = intfloor(fx)
        ibx = intfloor(bx)
        if ifx != ibx
            dvox = bx-fx
            sign = ifx < ibx ? +1 : -1
            crossings = ifx+sign:sign:ibx
            αx = (crossings .- fx) / dvox
        end

        # Find all crossings of y-plane boundaries
        fy = vfront.y[i]
        by = vback.y[i]
        ify = intfloor(fy)
        iby = intfloor(by)
        if ify != iby
            dvox = by-fy
            sign = ify < iby ? +1 : -1
            crossings = ify+sign:sign:iby
            αy = (crossings .- fy) / dvox
        end
        
        # Merge these together, in least-greatest order
        α = merge_sorted(αz, αx, αy)
        length(α) ≤ 1 && continue

        xyz_front = vfront[i]
        xyz_displ = dv[i]
        TD = total_distance[i]
        prevα = α[1]
        for nextα in α[2:end]
            thisdist = TD*abs(nextα-prevα)
            thisdist ≤ 0 && continue
            voxelID = [1+intfloor(x) for x in xyz_front + xyz_displ*prevα]
            any(voxelID .< 1) && continue
            any(voxelID .> v.voxelsPerEdge) && continue
            integral[i] += thisdist * v.densities[voxelID...]
            prevα = nextα
        end

        # Try sampling
        # stepsize = 0.02  # i.e., 20 nm
        # d = .5*stepsize:stepsize:total_distance[i]
        # ix = 1 .+int.(vfront.x[i] .+ dv.x[i]*d)
        # iy = 1 .+int.(vfront.y[i] .+ dv.y[i]*d)
        # iz = 1 .+int.(vfront.z[i] .+ dv.z[i]*d)
        # dsum = 0.0
        # for (a,b,c) in zip(ix, iy, iz)
        #     if a ≥ 1 && b ≥ 1 && c ≥ 1 && a ≤ v.voxelsPerEdge[1] && b ≤ v.voxelsPerEdge[2] && c ≤ v.voxelsPerEdge[3]
        #         dsum += v.densities[a, b, c]
        #     end
        # end
        # integral[i] = stepsize * dsum
    end
    integral
end