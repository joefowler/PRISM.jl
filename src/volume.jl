using LinearAlgebra

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