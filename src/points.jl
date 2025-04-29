using LinearAlgebra
using StaticArrays

struct Vec3D{T} <: FieldVector{3, T}
    x::T
    y::T
    z::T
end

StaticArrays.similar_type(::Type{<:Vec3D}, ::Type{T}, s::Size{(3,)}) where {T} = Vec3D

"""
    PointArray

Describes an ordered set of locations in 3d space. Uses 
"""
PointArray{T<:Real} = Vector{Vec3D{T}}

function PointArray(x::AbstractMatrix{T}) where T<:Real
    sm = [Vec3D(col...) for col in eachcol(x)]
    PointArray{T}(sm)
end
PointArray(x::AbstractVector, y::AbstractVector, z::AbstractVector) = PointArray(vcat(x', y', z'))
PointArray(x::Real, y::AbstractVector, z::AbstractVector) = PointArray(fill(x, length(y)), y, z)
PointArray(x::AbstractVector, y::Real, z::AbstractVector) = PointArray(x, fill(y, length(x)),z)
PointArray(x::AbstractVector, y::AbstractVector, z::Real) = PointArray(x, y, fill(z, length(y)))
PointArray(x::Real, y::Real, z::AbstractVector) = PointArray(fill(x, length(z)), fill(y, length(z)), z)
PointArray(x::AbstractVector, y::Real, z::Real) = PointArray(x, fill(y, length(x)),fill(z, length(x)))
PointArray(x::Real, y::AbstractVector, z::Real) = PointArray(fill(x, length(y)), y, fill(z, length(y)))
PointArray(x::Real, y::Real, z::Real) = PointArray([x], [y], [z])
PointArray(x::Matrix, y::Matrix, z::Matrix) = PointArray(vec(x), vec(y), vec(z))

LinearAlgebra.norm(p::PointArray, q::Real=2) = [norm(x, q) for x in p]
