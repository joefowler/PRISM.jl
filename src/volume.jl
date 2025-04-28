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
    n = length(pa1)
    enter = PointArray{Float64}(Array{Vec3D{Float64}}(undef, n))
    exit = PointArray{Float64}(Array{Vec3D{Float64}}(undef, n))

    mx, my, mz = minedges(v)
    qx, qy, qz = maxedges(v)
    for i = 1:n
        p = pa1[i]
        Δp = pa2[i] - p

        a = (mx-p.x)/Δp.x
        b = (qx-p.x)/Δp.x
        αxmin, αxmax = min(a,b), max(a,b)

        a = (my-p.y)/Δp.y
        b = (qy-p.y)/Δp.y
        αymin, αymax = min(a,b), max(a,b)

        a = (mz-p.z)/Δp.z
        b = (qz-p.z)/Δp.z
        αzmin, αzmax = min(a,b), max(a,b)

        αmin = max(αxmin, αymin, αzmin)
        αmax = min(αxmax, αymax, αzmax)
        if αmin > αmax
            αmin = αmax = 0.0
        end
        enter[i] = p + Δp*αmin
        exit[i] = p + Δp*αmax
    end
    enter, exit
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
    pfirst, plast = enter_exit_points(v, p1, p2)
    integral = zeros(Float64, length(p1))
    densities = v.densities

    vfirst = real2voxel(v, pfirst)
    vlast = real2voxel(v, plast)
    dv = vlast - vfirst
    vpex, vpey, vpez = v.voxelsPerEdge

    for i in 1:length(p1)
        vf = vfirst[i]
        vb = vlast[i]
        distanceInVolume = norm(pfirst[i]-plast[i])
        distanceInVolume ≤ 0 && continue

        # Let α=[0,1] represent [pfirst,plast]. Now find all values of α where a pixel boundary is crossed
        # Start with all crossings of x-plane boundaries, then y, then z
        α = [0.0, 1.0]  # In case, due to roundoff errors, these are not selected.
        for (f, b) in zip(vf, vb)
            iff = floor(Int, f)
            ibb = floor(Int, b)
            xyzcrossings = iff < ibb ? (iff+1:ibb) : (iff:-1:ibb+1)
            α = merge_sorted(α, (xyzcrossings .- f) / (b-f))
        end
        
        length(α) ≤ 1 && continue

        XF, YF, ZF = vfirst[i]
        XD, YD, ZD = dv[i]
        integral_i = 0.0
        prevα = α[1]
        for nextα in α[2:end]
            nextα == prevα && continue

            # Choose the pixel ID for this segment based on the midpoint between where it enters and
            # exits the pixel (that will be a fraction `midα` along the way from the "front" to the "back"
            # of the line segment). This prevents roundoff errors that might convert index 3.99999999 → 3
            # instead of the desired 4, if we used prevα or nextα alone as the indicator.
            midα = 0.5*(nextα+prevα)
            ix = floor(Int, XF + XD*midα)
            iy = floor(Int, YF + YD*midα)
            iz = floor(Int, ZF + ZD*midα)
            if ix ≥ 0 && iy ≥ 0 && iz ≥ 0 && ix < vpex && iy < vpey && iz < vpez
                index_1d = 1+ix + vpex*(iy + vpey*iz)
                integral_i += (nextα-prevα) * densities[index_1d]
            end
            prevα = nextα
        end
        integral[i] = distanceInVolume*integral_i
    end
    integral
end

path_integrated_density(v::Volume, p1, p2) = path_integrated_density(v, PointArray(p1...), PointArray(p2...))[1]

function path_integrated_density_fast(v::Volume, pa1::PointArray, pa2::PointArray)
    N = length(pa1)
    integral = Array{Float64}(undef, N)

    nx, ny, nz = v.voxelsPerEdge
    mx, my, mz = minedges(v)
    qx, qy, qz = maxedges(v)
    dx, dy, dz = stepsize(v)

    function x2voxel(p::AbstractArray)
        i = 1+floor(Int, (p[1]-mx)/dx)
        j = 1+floor(Int, (p[2]-my)/dy)
        k = 1+floor(Int, (p[3]-mz)/dz)
        i, j, k
    end
    voxel2index(i::Integer, j::Integer, k::Integer) = i + nx*((j-1) + (k-1)*ny)

    for i = 1:N
        p1 = pa1[i]
        p2 = pa2[i]
        p2 == p1 && continue
        Δp = p2 - p1
        signx = sign(Δp.x)
        signy = sign(Δp.y)
        signz = sign(Δp.z)

        # Find where ray enters/exits the x, y, then z limits of the rectangular region.
        a = (mx-p1.x)/Δp.x
        b = (qx-p1.x)/Δp.x
        αxmin, αxmax = min(a,b), max(a,b)

        a = (my-p1.y)/Δp.y
        b = (qy-p1.y)/Δp.y
        αymin, αymax = min(a,b), max(a,b)

        a = (mz-p1.z)/Δp.z
        b = (qz-p1.z)/Δp.z
        αzmin, αzmax = min(a,b), max(a,b)

        # Find α value where ray enters and leaves the region.
        # When αmin ≥ αmax, the ray does not actually cross the region.
        αmin = max(αxmin, αymin, αzmin)
        αmax = min(αxmax, αymax, αzmax)
        αmin ≥ αmax  && continue

        enter = p1 + Δp*αmin
        exit = p1 + Δp*αmax

        # Find size of the steps in α that span exactly one x, y, or z voxel.
        # These are positive even if ray has a negative x-, y-, or z- direction vector.
        # These may be infinite if ray is parallel to one or two axes--that's okay.
        Δαx = dx / abs(Δp.x)
        Δαy = dy / abs(Δp.y)
        Δαz = dz / abs(Δp.z)

        # Compute voxel numbers where ray enters and exits the region.
        ie, je, ke = x2voxel(enter)
        ix, jx, kx = x2voxel(exit)

        # Handle details that depend upon the last plane entered / first plane exited.
        # The assumption is that if ray enters on an edge (thus, 2 planes at once), we can ignore that.
        # If enter at (say) a y-plane, the je index is subject to roundoff error. Fix that.
        # N.B. Compute  (Δαz / signz) and similar, instead of multiplying. Point is so that a Δp.z==0 
        # will preserve Inf / 0 → Inf in nextαz. Whereas, Inf * 0 = NaN, which we don't want.
        if αmin == αxmin  # enter on a x-plane
            ie = Δp.x > 0 ? 1 : nx
            nextαx = αmin + Δαx
            nextαy = αmin + (Δαy / signy) * (je - (enter.y-my)/dy)
            nextαz = αmin + (Δαz / signz) * (ke - (enter.z-mz)/dz)
            if Δp.y < 0
                nextαy += Δαy
            end
            if Δp.z < 0
                nextαz += Δαz
            end
        elseif αmin == αymin  # enter on a y-plane
            je = Δp.y > 0 ? 1 : ny
            nextαy = αmin + Δαy
            nextαx = αmin + (Δαx / signx) * (ie - (enter.x-mx)/dx)
            nextαz = αmin + (Δαz / signz) * (ke - (enter.z-mz)/dz)
            if Δp.x < 0
                nextαx += Δαx
            end
            if Δp.z < 0
                nextαz += Δαz
            end
        else  # enter on a z-plane
            ke = Δp.z > 0 ? 1 : nz
            nextαz = αmin + Δαz
            nextαx = αmin + (Δαx / signx) * (ie - (enter.x-mx)/dx)
            nextαy = αmin + (Δαy / signy) * (je - (enter.y-my)/dy)
            if Δp.x < 0
                nextαx += Δαx
            end
            if Δp.y < 0
                nextαy += Δαy
            end
        end

        if αmax == αxmax  # exit from an x-plane
            ix = Δp.x < 0 ? 1 : nx
        elseif αmax == αymax  # exit from a y-plane
            jx = Δp.y < 0 ? 1 : ny
        else  # exit from a z-plane
            kx = Δp.z < 0 ? 1 : nz
        end

        Δi = Δp.x > 0 ? 1 : -1
        Δj = Δp.y > 0 ? nx : -nx
        Δk = Δp.z > 0 ? ny*nx : -ny*nx

        # The variables that will update in each voxel crossed
        running_sum = 0.0
        α = αmin
        index = voxel2index(ie, je, ke)
        nVoxelsCrossed = abs(ix-ie) + abs(jx-je) + abs(kx-ke) + 1
        for j = 1:nVoxelsCrossed
            if nextαx < min(nextαy, nextαz)
                # An x step is shortest
                index_step = Δi
                nextα = nextαx
                nextαx += Δαx
            elseif nextαy < nextαz
                # A y step is shortest
                index_step = Δj
                nextα = nextαy
                nextαy += Δαy
            else
                # A z step is shortest
                index_step = Δk
                nextα = nextαz
                nextαz += Δαz
            end

            # To avoid illegal access outside array v.densities, skip any 0-length line segments
            # (especially those at the end of the ray-volume overlap).
            if nextα > α
                running_sum += (nextα - α) * v.densities[index]
            end
            α = nextα
            index += index_step
        end
        integral[i] = running_sum * norm(Δp)
    end
    integral
end
path_integrated_density_fast(v::Volume, p1, p2) = path_integrated_density_fast(v, PointArray(p1...), PointArray(p2...))[1]
