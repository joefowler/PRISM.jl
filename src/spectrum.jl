using Dierckx  # wraps a Fortran spline library
using NeXLCore  # x-ray data
using LinearAlgebra
using QuadGK
using Printf
using Statistics
using HDF5

    """_summary_
    """
struct Spectrum
    continuum
    lorentz_energies::Vector
    lorentz_halfwidths::Vector
    lorentz_intensities::Vector
end

Spectrum(continuum) = Spectrum(continuum, [], [], [])

function Spectrum_example(include_platinum=true)
    e = [5, 6.2, 7.5, 10, 13, 16, 20, 22, 25]*1e3
    y = [.4462, .482, .47, .385, .279, .203, .133, .102, 0]
    spline = Spline1D(e, y)
    if include_platinum
        energies = [8268, 9362, 9442.4, 9975, 10854.4, 11070.8, 11230.9, 11233, 11250.7, 11557, 12942, 13310]
        Γ = 0.5 * [24, 7, 7, 24, 26, 8, 25, 13, 12, 50, 13, 100]
        intensities = [2.91, 5.6, 49.69, .5, 1.9, 20.06, 1.42, 1.13, 10.13, .91, 4.23, 1.02] * 1.160118
    else
        energies, Γ, intensities = [], [], []
    end
    Spectrum(spline, energies, Γ, intensities)
end

function (S::Spectrum)(energy::AbstractArray)
    result = S.continuum(energy)
    if length(S.lorentz_energies) > 0
        for (Epeak, Γ, I0) in zip(S.lorentz_energies, S.lorentz_halfwidths, S.lorentz_intensities)
            lorentz = I0 ./ (1 .+ ((energy.-Epeak) ./ Γ).^2)
            result .+= lorentz
        end
    end
    result
end
(S::Spectrum)(energy::Real) = S([energy])[1]

sio2 = parse(Material, "SiO2", density=2.65)
cu = pure(n"Cu")
si = pure(n"Si")
be = pure(n"Be")
graphite = pure(n"C")

# mac() returns mass attenuation in cm^2/g; multiply by density (in g/cm^3) to get attenuation per cm:
atten_per_cm(m::Material, e::Float64) = mac(m, e) * density(m)
atten_per_cm(m::Material, e::Real) = mac(m, float(e)) * density(m)
atten_per_cm(m::Material, evec::AbstractVector) = [mac(m, float(e)) for e in evec] * density(m)
transmission(m, e, thickness::Real) = exp.(-atten_per_cm(m, e) * thickness)

"""
    total_efficiency(energy; secσ=1.0, secκ=1.0)

Return the total efficiency of our tomography sample given loss in SiO2, graphite, Be window,
and incomplete absorption in the silicon camera at given `energy` (a vector or single value).

Arguments
---------
- `secσ` is the secant of the angle σ between a ray and the sample sandwich. 
    Rays see the SiO2 and graphite layers elongated by this factor.
- `secκ` is the secant of the angle κ between a ray and the x-ray camera. 
    Rays see the beryllium window and the camera detector wafer elongated by this factor.
"""
function total_efficiency(energy::AbstractVector; secσ=1.0, secκ=1.0)
    QE = 1 .- transmission(si, energy, 450e-4 * secκ)
    QE .*= transmission(be, energy, 200e-4 * secκ)
    QE .*= transmission(graphite, energy, 60e-4 * secσ)
    QE .*= transmission(sio2, energy, 3.3e-4 * secσ)
    QE
end
total_efficiency(energy::Real; kargs...) = total_efficiency([energy]; kargs...)[1]


function compute_H(take_derivative=false)
    K = LinRange(1, 1.091, 6)
    S = LinRange(1, 2.2074, 10)
    L = LinRange(0, 4e-4, 11)
    R = Float64[]
    x = Float64[]
    y = Float64[]
    z = Float64[]
    spectrum = Spectrum_example()
    for k in K
        for s in S
            for λ in L
                @show k,s,λ
                if take_derivative
                    g = quadgk(x->spectrum(x)*total_efficiency(x; secκ=k, secσ=s)*transmission(cu, x, λ)*atten_per_cm(cu, x),
                                5e3, 25e3)[1]
                    push!(R, g)
                else
                    h = quadgk(x->spectrum(x)*total_efficiency(x; secκ=k, secσ=s)*transmission(cu, x, λ), 5e3, 25e3)[1]
                    push!(R, h)
                end
                push!(x, λ*1e4)
                push!(y, k)
                push!(z, s)
            end
        end
    end
    return R, K, S, L, x, y, z
end

function matching_pursuit_H_fit(sampledF, x, y, zz)
    M = []
    powers = []
    Use = Bool[]
    MAXPOWER=3
    for total=0:MAXPOWER
        for j=0:total
            for i=0:total-j
                k = total - (i+j)
                vec = x.^i .* y.^j .* zz.^k
                vec /= norm(vec)
                push!(M, vec)
                push!(powers, (i,j,k))
                push!(Use, total ≤ 2)
            end
        end
    end
    M = hcat(M...)
    param = M\sampledF

    nparam = size(M)[2]
    param = M[:, Use] \ sampledF
    residual = sampledF - M[:, Use]*param
    goal = .3e-4
    @show powers[Use]
    @show maximum(abs.(residual))

    for iter=1:15
        dd = M'*residual
        # @show abs.(dd)/maximum(abs.(dd))
        winner = argmax(abs.(dd))
        Use[winner] = true
        param = M[:, Use] \ sampledF
        residual = sampledF - M[:, Use]*param
        # residual .-= M[:, winner]*dd[winner]
        @show sum(Use), winner, powers[winner], maximum(abs.(residual))
        maximum(abs.(residual)) < goal && break
    end
    param = M[:, Use] \ sampledF
end


function final_fit(G, H, λ, secκ, secσ)
    M = hcat(ones(Float64, length(G)), λ, λ.^2, λ.^3, secκ, secκ.^2, secσ, secσ.^2, λ.*secκ, λ.*secσ, secκ.*secσ)
    param_logH = M \ log.(H)
    resid = log.(H) - M*param_logH
    rms = x->sqrt(mean(x.^2))
    @printf "Log(H) absolute residual rms %.3e  max abs %.3e\n" rms(resid) maximum(abs.(resid))

    GH = G./H
    param_GH = M \ GH
    resid = GH - M*param_GH
    @printf "G/H    absolute residual rms %.3e  max abs %.3e\n" rms(resid) maximum(abs.(resid))
    @printf "G/H    relative residual rms %.3e  max abs %.3e\n" rms(resid)/GH[1] maximum(abs.(resid))/GH[1]
    param_logH, param_GH
end


function store_GH_integrals(filename::AbstractString)
    G, _, _, _, λ, secκ, secσ = compute_H(true)
    H, _, _, _, λ, secκ, secσ = compute_H(false)
    param_logH, param_GH = final_fit(G, H, λ, secκ, secσ)

    h5open(filename, "w") do h5
        h5["logH"] = param_logH
        h5["G_over_H"] = param_GH
    end
end