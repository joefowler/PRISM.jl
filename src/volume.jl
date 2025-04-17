using LinearAlgebra

    """PointArray

Describes an ordered set of locations in 3d space. 

Fields
---------
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    n::Int

    """
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

    """Volume

Represent a volume broken into voxels shaped as rectangular prism.

Fields
-------
    edges         :: T<:(AbstractVector)
    densities     :: Array{Float64}
    voxelsPerEdge :: Vector{Int64}
    voxelsTotal   :: Int64
    """
struct Volume{T<:AbstractVector}
    edges::T
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

function real2voxel(v::Volume, pa::PointArray)
    ex, ey, ez = v.edges
    x = (pa.x .- minimum(ex)) / step(ex)
    y = (pa.y .- minimum(ey)) / step(ey)
    z = (pa.z .- minimum(ez)) / step(ez)
    PointArray(x, y, z, pa.n)
end
real2voxel(v::Volume, p::AbstractArray) = [(x-minimum(e))/step(e) for (x,e) in zip(p, v.edges)]

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

    """merge_sorted(a::AbstractVector, b::AbstractVector, [c::AbstractVector])

    Assumes that both `a` and `b` are pre-sorted vectors of the same type.
    Returns an array that contains all elements of either input, sorted. If `c` is also given, merge all 3 arrays.
    """
function merge_sorted(a::AbstractVector, b::AbstractVector)
    na = length(a)
    nb = length(b)
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
merge_sorted(a::AbstractVector, b::AbstractVector, args...) = merge_sorted(merge_sorted(a, b), args...)

function path_integrated_density(v::Volume, p1::PointArray, p2::PointArray)
    pfront, pback = enter_exit_points(v, p1, p2)
    integral = zeros(Float64, p1.n)
    total_distance = norm(pback-pfront)
    densities = v.densities

    vfront = real2voxel(v, pfront)
    vback = real2voxel(v, pback)
    dv = vback - vfront
    intfloor(x::Real) = floor(Int, x)
    intround(x::Real) = round(Int, x)
    vpex, vpey, vpez = v.voxelsPerEdge

    for i in 1:p1.n
        # Let α=[0,1] represent [pfront,pback]. Now find all values of α where a pixel boundary is crossed
        # Start with all crossings of z-plane boundaries
        α = LinRange(0, 1, 1+v.voxelsPerEdge[3])

        # Find all crossings of x-plane boundaries
        fx = vfront.x[i]
        bx = vback.x[i]
        ifx = intfloor(fx)
        ibx = intfloor(bx)
        xcrossings = ifx < ibx ? (ifx+1:ibx) : (ifx:-1:ibx+1)
        α = merge_sorted(α, (xcrossings .- fx) / (bx-fx))

        # Find all crossings of y-plane boundaries
        fy = vfront.y[i]
        by = vback.y[i]
        ify = intfloor(fy)
        iby = intfloor(by)
        ycrossings = ify < iby ? (ify+1:iby) : (ify:-1:iby+1)
        α = merge_sorted(α, (ycrossings .- fy) / (by-fy))
        
        length(α) ≤ 1 && continue

        TD = total_distance[i]
        XF, YF, ZF = vfront[i]
        XD, YD, ZD = dv[i]
        integral_i = 0.0
        prevα = α[1]
        for nextα in α[2:end]
            # Choose the pixel ID for this segment based on the midpoint between where it enters and
            # exits the pixel (that will be a fraction `midα` along the way from the "front" to the "back"
            # of the line segment). This prevents roundoff errors that might convert index 3.99999999 → 3
            # instead of the desired 4, if we used prevα or nextα alone as the indicator.
            midα = 0.5*(nextα+prevα)
            ix = intfloor(XF + XD*midα)
            iy = intfloor(YF + YD*midα)
            iz = intfloor(ZF + ZD*midα)
            if ix ≥ 0 && iy ≥ 0 && iz ≥ 0 && ix < vpex && iy < vpey && iz < vpez
                thisdist = TD*abs(nextα-prevα)
                index_1d = 1+ix + vpex*(iy + vpey*iz)
                integral_i += thisdist * densities[index_1d]
            end
            prevα = nextα
        end
        integral[i] = integral_i
    end
    integral
end

path_integrated_density(v::Volume, p1, p2) = path_integrated_density(v, PointArray(p1), PointArray(p2))[1]
