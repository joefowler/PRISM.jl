#!/usr/bin/env julia
using Pkg
Pkg.activate(dirname(@__DIR__))
Pkg.resolve() #Optional if a known-good Manifest.toml is included
Pkg.instantiate()

using Printf
using HDF5
using YAML
using Glob
using Statistics
using PRISM
import .PRISM: downsample_2d, Image, Camera

mutable struct Configure
    rescale::Int
    maxtriggers::Int
    dryrun::Bool
    clobber::Bool
    output::String
    inputs::Vector{String}
    angles::Vector{Float64}
end

function load_images(h::HDF5.H5DataStore, rescale::Integer=1; maxtriggers=-1)
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
    raw_eds = reshape(pp["eds/fitted_peak_heights"][:], nimages, ntrigger)
    eds = dropdims(mean(raw_eds, dims=1), dims=1)/mean(raw_eds)

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
    if maxtriggers ≤ 1
        maxtriggers = ntrigger
    end
    println("...about to read $(maxtriggers) images from $(h.filename)")
    for i = 1:maxtriggers
        imgdata = dropdims(sum(data[:,:,1,1+nimages*(i-1):nimages*i]; dims=3); dims=3)
        imgdata[mask .> 0] .= 0
        rawimg = downsample_2d(imgdata, rescale; outputtype=UInt32)
        Img = Image(rawimg, goodPix, eds[i], frame_time*nimages, nimages, camera)
        push!(images, Img)
        println("...read image $(i) with ΔT=$(frame_time*nimages) and intensity $(eds[i])")
    end
    camera, images, positions
end

function load_images(hdf5file::AbstractString, rescale::Integer=1; maxtriggers=-1) 
    h5open(hdf5file, "r") do h
        return load_images(h, rescale; maxtriggers=maxtriggers)
    end
end

HOMEsubstitute(s::AbstractString) = replace(s, raw"$HOME"=>homedir())

function learn_configuration(filename="default_file.h5")
    # load & preprocess your data here
    configdata = YAML.load_file(filename)

    @assert haskey(configdata, "output")
    @assert haskey(configdata, "input")
    rescale = get(configdata, "rescale", 1)
    maxtriggers = get(configdata, "maxtriggers", -1)
    dryrun = get(configdata, "dryrun", false)
    clobber = get(configdata, "clobber", false)

    output = configdata["output"]
    output = HOMEsubstitute(output)

    input_root = configdata["input_root"]
    input_root = HOMEsubstitute(input_root)

    inputs = String[]
    angles = Float64[]
    for (globpattern, a) in configdata["input"]
        files = glob(HOMEsubstitute(globpattern), input_root)
        append!(inputs, files)
        append!(angles, fill(a, length(files)))
    end

    Configure(rescale, maxtriggers, dryrun, clobber, output, inputs, angles)
end

function report_plan(config::Configure)
    println("rescale_raw_data.jl plan:")
    println("=========================")
    println("Rescale pixel factor: $(config.rescale)")
    if isfile(config.output)
        if config.clobber
            println("Writing output to:    $(config.output) (overwriting because clobber=true)")
        else
            println("Writing output to:    $(config.output) (but will error because clobber=false)")
        end
    else
        println("Writing output to:    $(config.output) (a new file)")
    end

    if config.maxtriggers ≥ 1
        println("Will limit output to $(config.maxtriggers) dwells per input file")
    else
        println("Will output all dwell positions")
    end
    for (f, a) in zip(config.inputs, config.angles)
        println("Angle $(a) degrees: $(f)")
    end
    println("=========================")
    if config.dryrun
        println("(dry run requested...quitting now)")
    end
end

function workhorse_function(config)
    config.dryrun && println("(dry run requested...quitting now)")

    if isfile(config.output)
        if config.clobber
            rm(config.output)
        else
            throw(ErrorException("Cannot write to $(config.output), it already exists"))
        end
    end

    positions = []
    intensities = Float64[]
    integrationTime = Float64[]
    angles = []

    h5open(config.output, "w") do h
        images_group = create_group(h, "images")
        image_ID = 1
        for (inputFile, angle) in zip(config.inputs, config.angles)
            camera, images, xyz = load_images(inputFile, config.rescale; maxtriggers=config.maxtriggers)
            if !("camera" in keys(h))
                cg = create_group(h, "camera")
                cg["pixel_centers"] = camera.pixelCenters
            end
            push!(positions, xyz)
            push!(angles, fill(Float64(angle), length(images)))

            for img in images
                gname = @sprintf "image%04d" image_ID
                g = create_group(images_group, gname)
                g["counts"] = img.counts
                g["validPix"] = img.validPix
                image_ID += 1
            end
            append!(intensities,[i.intensityScale for i in images])
            append!(integrationTime, [i.integrationTime for i in images])
        end
        positions = hcat(positions...)
        angles = vcat(angles...)
        h["angles"] = angles
        h["sample_position"] = positions
        h["sourceIntensities"] = intensities
        h["integrationTimes"] = integrationTime
    end
end

function main()
    if length(ARGS) != 1
        throw(ArgumentError("Must run $(@__FILE__) with a YAML configuration file as command-line argument."))
    end
    config_fname = ARGS[1]
    config = learn_configuration(config_fname)
    report_plan(config)
    workhorse_function(config)
end

!isinteractive() && main() # this line is where the magic happens
