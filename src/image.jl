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

function build_radiograph(camera::Camera, images::AbstractVector{Image}, positions::AbstractMatrix)
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
