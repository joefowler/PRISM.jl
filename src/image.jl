using HDF5

struct Image
    counts::Matrix{UInt32}
    validPix::Matrix{UInt8}
    intensityScale::Float64
    integrationTime::Float64
    nimages::Int
    camera::Camera
end

function load_images(h::HDF5.H5DataStore, rescale::Integer=1)
    data = h["entry/data/data"]
    pp = h["entry/post_processed"]
    nimages = pp["detector_info/nimages"][1]  # Expect 6, 3, or 2.
    ntrigger = pp["detector_info/ntrigger"][1]
    positions = pp["dwell_positions/corrected_positions"]
    @assert attrs(positions)["coords"] == "[X,Y,Z]"
    @assert attrs(positions)["units"] == "um"
    positions = positions[:,:]
    detector = h["entry/instrument/detector"]
    c1 = detector["threshold_1_channel"]
    mask = c1["pixel_mask"][:,:]
    mask[1407, 1565] = 0xff
    mask[421, 1539] = 0xff
    # threshold_energy = read(c1["threshold_energy"])
    frame_time = read(detector["frame_time"])
    eds = pp["eds/fitted_peak_heights"][:]
    eds *= 1/mean(eds)

    # Check that pixel sizes are given in meters, then convert to µm.
    Δx = read(detector["x_pixel_size"])
    Δy = read(detector["y_pixel_size"])
    @assert attrs(detector["x_pixel_size"])["units"] == "m"
    @assert attrs(detector["y_pixel_size"])["units"] == "m"
    Δx *= 1e6
    Δy *= 1e6

    nx, ny, ntriglevels, nframes = size(data)
    @assert ntriglevels == 1
    @assert nframes == nimages*ntrigger
    x = Δx * (-0.5*(nx-1):0.5*(nx-1))
    y = Δy * (-0.5*(ny-1):0.5*(ny-1))
    rawpix_is_good = mask .== 0
    goodPix = downsample_2d(rawpix_is_good, rescale; outputtype=UInt8)

    X = vec(x)*ones(length(y))'
    Y = ones(length(x))*vec(-y)'
    xctr = downsample_2d(X.*rawpix_is_good, rescale) ./ goodPix
    yctr = downsample_2d(Y.*rawpix_is_good, rescale) ./ goodPix
    pixCtrs = PointArray(vec(xctr), vec(yctr), 257500.0)
    camera = Camera(pixCtrs)
    images = Image[]
    for i = 1:ntrigger
        imgdata = dropdims(sum(data[:,:,1,1+nimages*(i-1):nimages*i]; dims=3); dims=3)
        imgdata[mask .> 0] .= 0
        rawimg = downsample_2d(imgdata, rescale; outputtype=UInt32)
        Img = Image(rawimg, goodPix, eds[i], frame_time*nimages, nimages, camera)
        return Img
        push!(images, Img)
    end

    camera, images, positions
end

function load_images(hdf5file::AbstractString, rescale::Integer=1) 
    h5open(hdf5file, "r") do h
        return load_images(h, rescale)
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

