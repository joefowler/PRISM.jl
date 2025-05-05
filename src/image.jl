using HDF5
using Statistics


struct Image
    counts::Matrix{UInt32}
    validPix::Matrix{UInt8}
    intensityScale::Float64
    integrationTime::Float64
    nimages::Int
    camera::Camera
end

function load_images(h::HDF5.H5DataStore)
    xyz = h["camera/pixel_centers"][:]
    Vxyz = Vec3D.(xyz)
    pixCtrs = PointArray{typeof(xyz[1].x)}(Vxyz)
    camera = Camera(pixCtrs)

    integrationTimes = h["integrationTimes"]
    sample_positions = h["sample_position"][:,:]
    intensities = h["sourceIntensities"]
    θ = h["angles"][:]
    ntrigger = minimum([length(x) for x in (integrationTimes, sample_positions, intensities, θ)])

    images = Vector{Image}(undef, ntrigger)
    for i = 1:ntrigger
        gname = @sprintf "images/image%04d" i
        g = h[gname]
        counts = g["counts"][:,:]
        validPix = g["validPix"][:,:]
        T = integrationTimes[i]
        nimages = round(Int, T/10.0)
        Img = Image(counts, validPix, intensities[i], T, nimages, camera)
        images[i] = Img
    end

    camera, images, sample_positions, θ
end

function load_images(hdf5file::AbstractString) 
    h5open(hdf5file, "r") do h
        return load_images(h)
    end
end

function downsample_2d(a::AbstractMatrix, rescale=1; outputtype=nothing)
    rescale ≤ 1 && return a
    nx, ny = size(a)
    rnx = div(nx+rescale-1, rescale)
    rny = div(ny+rescale-1, rescale)

    if outputtype === nothing
        imax = min(nx*ny, 8)
        outputtype = typeof(sum(a[1:imax]))
    end

    out = zeros(outputtype, rnx, rny)
    for iy = 1:rny
        iy0 = 1+rescale*(iy-1)
        iy9 = min(rescale*iy, ny)
        a_cols = a[:,iy0:iy9]
        for ix = 1:rnx
            ix0 = 1+rescale*(ix-1)
            ix9 = min(rescale*ix, nx)
            a_area = a_cols[ix0:ix9, :]
            out[ix, iy] = sum(a_area)
        end
    end
    out
end

function build_radiograph(camera::Camera, images::AbstractVector{Image}, positions::AbstractMatrix; pixelReject=(x->false))
    @assert length(images) == size(positions)[2]
    nominal_z = 10.0
    pixCtrs = camera.pixelCenters*(nominal_z/camera.pixelCenters[1].z)
    
    xedges = 81:.05:123
    yedges = -21:.05:-3.5
    nx = length(xedges)
    ny = length(yedges)
    mx = minimum(xedges)
    my = minimum(yedges)
    dx = step(xedges)
    dy = step(yedges)
    obliquity = camera.secκ.^-3


    bigw = zeros(Float64, ny, nx)
    bigc = zeros(Float64, ny, nx)
    for (i,img) in enumerate(images)
        offset = positions[:,i]
        counts = vec(img.counts)
        npix = vec(img.validPix)
        @assert length(pixCtrs) == length(counts)
        @assert length(pixCtrs) == length(npix)
        for (c,w,pc,obliq) in zip(counts, npix, pixCtrs, obliquity)
            w ≤ 0 && continue
            pixelReject(pc) && continue
            w *= img.intensityScale * img.integrationTime * obliq
            x, y, _ = pc + offset
            ix = floor(Int, (x-mx)/dx)
            iy = floor(Int, (y-my)/dy)
            (ix < 1 || ix > nx) && continue
            (iy < 1 || iy > ny) && continue
            bigw[iy, ix] += w
            bigc[iy, ix] += c
        end
    end
    bigc ./ bigw
end

function build_all_radiographs(camera::Camera, images::AbstractVector{Image}, positions::AbstractMatrix, angles::AbstractVector)
    results = Dict{Float64,Matrix{Float64}}()
    uniqueA = sort(unique(angles))
    for θ in uniqueA
        use = angles .== θ
        offset = [10tan(θ*π/180), 0, 0]
        rg = build_radiograph(camera, images[use], positions[:,use].-offset)
        results[θ] = rg
    end
    results
end

function offsets_feb_2025()
    # The number of inner, middle, outer dwells for θ=[0, 7.5, 15, ...45, -7.5, -15, ... -37.5]
    nframes = reshape([
        72, 27, 24, 76, 30, 30, 76, 30, 30, 72, 30, 30, 76, 30, 27, 80, 30, 27, 84, 36, 27,
        76, 30, 30, 76, 30, 27, 72, 30, 30, 76, 30, 27, 80, 30, 27], 3, 12)
    angleorder = vcat(0:7.5:45, -7.5:-7.5:-37.5)
    corrections = [
        0       0         0         0         0           0         0
        -37.5   2.368    -0.31875   1.785    -0.7        1.30633   -0.929333
        -30.0   3.66425  -0.82875   3.37733  -1.28667    2.621     -1.96367
        -22.5   3.9235   -0.7225    3.794    -1.10533    3.117     -1.482
        -15.0   3.388    -0.9945    3.27533  -1.45667    2.62967   -2.19333
         -7.5   2.3595   -0.646     2.23833  -1.074      1.621     -1.51333
          7.5  -0.561    -0.15725  -0.561    -0.209667  -0.586667  -0.547
         15.0  -1.76375  -0.238    -1.785    -0.178333  -1.87033   -0.445
         22.5  -2.5755   -0.051    -2.62933  -0.014     -2.92433   -0.187
         30.0  -4.59425   0.13175  -4.68367   0.269333  -4.04367    0.824667
         37.5  -6.04775  -0.25925  -5.817     0.261     -5.024      1.026
         45.0  -8.386     0.306    -8.18667   0.802333  -6.55967    2.1
       
    ]
    xinner = Dict(corrections[:, 1] .=> corrections[:, 2])
    yinner = Dict(corrections[:, 1] .=> corrections[:, 3])
    xmiddle = Dict(corrections[:, 1] .=> corrections[:, 4])
    ymiddle = Dict(corrections[:, 1] .=> corrections[:, 5])
    xouter = Dict(corrections[:, 1] .=> corrections[:, 6])
    youter = Dict(corrections[:, 1] .=> corrections[:, 7])
    xoffsets = Float64[]
    yoffsets = Float64[]
    for (i, angle) in enumerate(angleorder)
        append!(xoffsets, fill(xinner[angle], nframes[1,i]))
        append!(yoffsets, fill(yinner[angle], nframes[1,i]))
        append!(xoffsets, fill(xmiddle[angle], nframes[2,i]))
        append!(yoffsets, fill(ymiddle[angle], nframes[2,i]))
        append!(xoffsets, fill(xouter[angle], nframes[3,i]))
        append!(yoffsets, fill(youter[angle], nframes[3,i]))
    end
    # Email May 2 says to reverse the sign of all y-offsets.
    sign_may2 = -1
    vcat(xoffsets', sign_may2*yoffsets', zeros(Float64, sum(nframes))')
end
