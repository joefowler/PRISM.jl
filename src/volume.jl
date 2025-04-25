using LinearAlgebra

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
minedges(v::Volume) = [minimum(x) for x in v.edges]
maxedges(v::Volume) = [maximum(x) for x in v.edges]
stepsize(v::Volume) = [step(a) for a in v.edges]

function real2voxel(v::Volume, pa::PointArray)
    mx, my, mz = minedges(v)
    sx, sy, sz = stepsize(v)
    m = [Vec3D((p.x-mx)/sx, (p.y-my)/sy, (p.z-mz)/sz) for p in pa]
    PointArray{eltype(m[1])}(m)
end
real2voxel(v::Volume, p::AbstractArray) = [(x-minimum(e))/step(e) for (x,e) in zip(p, v.edges)]

function enter_exit_points(v::Volume, p1::Vec3D, p2::Vec3D)
    Δp = p2 - p1

    mx, my, mz = minedges(v)
    qx, qy, qz = maxedges(v)
    αxmin = (mx - p1.x) / Δp.x
    αxmax = (qx - p1.x) / Δp.x
    αymin = (my - p1.y) / Δp.y
    αymax = (qy - p1.y) / Δp.y
    αzmin = (mz - p1.z) / Δp.z
    αzmax = (qz - p1.z) / Δp.z
    if αxmin > αxmax
        αxmin, αxmax = αxmax, αxmin
    end
    if αymin > αymax
        αymin, αymax = αymax, αymin
    end
    if αzmin > αzmax
        αzmin, αzmax = αzmax, αzmin
    end

    αmin = max(αxmin, αymin, αzmin)
    αmax = min(αxmax, αymax, αzmax)
    # If ray doesn't pass through the region, that's seen with αmin<αmax
    if αmax < αmin
        αmax = αmin
    end
    p1 + αmin*Δp, p1 + αmax*Δp
end
enter_exit_points(v::Volume, p1::AbstractArray, p2::AbstractArray) = enter_exit_points(v, Vec3D(p1), Vec3D(p2))

function enter_exit_points(v::Volume, pa1::PointArray, pa2::PointArray)
    Δp = pa2 - pa1
    n = length(pa1)
    αmin = Array{Float64}(undef, n)
    αmax = Array{Float64}(undef, n)

    mx, my, mz = minedges(v)
    qx, qy, qz = maxedges(v)
    for i = 1:n
        p = pa1[i]
        dp = Δp[i]

        a = (mx-p.x)/dp.x
        b = (qx-p.x)/dp.x
        αxmin, αxmax = a < b ? (a, b) : (b, a)

        a = (my-p.y)/dp.y
        b = (qy-p.y)/dp.y
        αymin, αymax = a < b ? (a, b) : (b, a)

        a = (mz-p.z)/dp.z
        b = (qz-p.z)/dp.z
        αzmin, αzmax = a < b ? (a, b) : (b, a)

        a = max(αxmin, αymin, αzmin)
        b = min(αxmax, αymax, αzmax)
        if a < b
            αmin[i] = a
            αmax[i] = b
        else
            αmin[i] = αmax[i] = -1.0
        end
    end
        
    pa1 + (Δp.*αmin), pa1 + (Δp.*αmax)
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
    integral = zeros(Float64, length(p1))
    total_distance = norm(pback-pfront)
    densities = v.densities

    vfront = real2voxel(v, pfront)
    vback = real2voxel(v, pback)
    dv = vback - vfront
    intfloor(x::Real) = floor(Int, x)
    intround(x::Real) = round(Int, x)
    vpex, vpey, vpez = v.voxelsPerEdge

    for i in 1:length(p1)
        # Let α=[0,1] represent [pfront,pback]. Now find all values of α where a pixel boundary is crossed
        # Start with all crossings of z-plane boundaries
        α = LinRange(0, 1, 1+v.voxelsPerEdge[3])

        # Find all crossings of x-plane boundaries
        fx = vfront[i].x
        bx = vback[i].x
        ifx = intfloor(fx)
        ibx = intfloor(bx)
        xcrossings = ifx < ibx ? (ifx+1:ibx) : (ifx:-1:ibx+1)
        α = merge_sorted(α, (xcrossings .- fx) / (bx-fx))

        # Find all crossings of y-plane boundaries
        fy = vfront[i].y
        by = vback[i].y
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

path_integrated_density(v::Volume, p1, p2) = path_integrated_density(v, PointArray(p1...), PointArray(p2...))[1]
