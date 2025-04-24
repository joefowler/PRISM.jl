using FFTW

"""
    lorentzSpotDFT(dx::Real, nx::Integer, dy::Real, ny::Integer, widthx::Real, [widthy::Real, θ::Real=0.0])

Return the discrete Fourier transform of size `(ny,nx)` for a smearing kernel shaped like 
`1 / (1 + (x/widthx)^2 + (y/widthy)^2)` for a sampled function with stepsize `(dy,dx)` in the y and
x directions.

If `widthy` is ommitted, widths are assumed equal in both directions. If `θ` is given a non-zero value
and the two widths are unequal, then the asymmetric smearing axes are rotated clockwise by that angle
(given in radians).

Use `real.(ifft(fft(image).*conj.(lorentzSpotDFT(...))))` to convolve `image` with the desired spot. (See convenience
function `lorentz_smearing`).
"""
function lorentzSpotDFT(dx::Real, nx::Integer, dy::Real, ny::Integer, widthx::Real, widthy::Real=-1, θ::Real=0.0)
    x = fftfreq(nx, dx*nx)
    y = fftfreq(ny, dy*ny)
    if widthx == 0
        x = fill(dx*nx*1e20, nx)
        x[1] = 0.0
    end
    if widthy < -1
        widthy = widthx
    end
    if widthy == 0
        y = fill(dy*ny*1e20, ny)
        y[1] = 0.0
    end

    Y = y * ones(nx)'
    X = ones(ny) * x'
    if (θ != 0) && (widthx != widthy)
        X, Y = X*cos(θ).-Y*sin(θ), X*sin(θ).+Y*cos(θ)
    end
    spot = 1. ./ (1 .+ (X./widthx).^2 .+ (Y./widthy).^2)
    fft(spot ./ sum(spot))
end

lorzentz_smearing(image::AbstractMatrix, x::Real, nx::Integer, dy::Real, ny::Integer, widthx::Real, widthy::Real=-1.0, θ::Real=0.0) =
    real.(ifft(image).*conj.(lorentzSpotDFT(dx, nx, dy, ny, widthx, widthy, θ)))
