using LinearAlgebra

struct Camera
    pixelCenters::PointArray
    secκ::Vector{Float64}
    secσ::Dict{Float64,Vector{Float64}}
    function Camera(centers)
        z = [p.z for p in centers]
        secκ = compute_secσ(centers, 0.0)
        secσ = Dict{Float64,Vector{Float64}}(0.0=>secκ)
        new(centers, secκ, secσ)
    end
end
Base.length(c::Camera) = length(c.pixelCenters)

function secσ(c::Camera, θ::Real)
    if !haskey(c.secσ, θ)
        c.secσ[θ] = compute_secσ(c, θ)
    end
    c.secσ[θ]
end

function compute_secσ(pixelCenters::PointArray, θ::Real)
    N = length(pixelCenters)
    secσ = Array{Float64}(undef, N)
    n = Vec3D(sin(θ), 0, cos(θ))
    for (i,pc) in enumerate(pixelCenters)
        secσ[i] = norm(pc) / dot(n, pc)
    end
    secσ
end
compute_secσ(c::Camera, θ::Real) = compute_secσ(c.pixelCenters, θ)