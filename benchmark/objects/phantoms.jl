const REGISTRY = JSON.parsefile(joinpath(@__DIR__, "registry.json"))
const SHAPES = JSON.parsefile(joinpath(@__DIR__, "shapes.json"))
const MATERIALS = Dict(k => Tuple(v) for (k,v) in SHAPES["materials"])
const ANATOMY = JSON.parsefile(joinpath(@__DIR__, "anatomy.json"))

function anatomical_phantom(ref; field_strength_T=nothing)
    if ref in ("cardiac_2d@1", "cardiac_3d@1")
        field = isnothing(field_strength_T) ? 3.0 : field_strength_T
        suffix = get(Dict(0.55=>"055t", 1.5=>"15t", 3.0=>"3t"), field, nothing)
        isnothing(suffix) && error("No cardiac tissue profile for scanner field strength")
        ref = replace(ref, "@1"=>"_" * suffix * "@1")
    end
    meta = ANATOMY["objects"][ref]
    profile = ANATOMY["field_profiles"][meta["field_profile"]]
    field_strength_T = isnothing(field_strength_T) ? profile["field_strength_T"] : field_strength_T
    field_strength_T == profile["field_strength_T"] || error("Cardiac tissue profile and scanner field strength must match")
    isabspath(meta["path"]) && error("Anatomical asset path must be relative")
    path = realpath(joinpath(@__DIR__, meta["path"]))
    relative = relpath(path, realpath(@__DIR__))
    (relative == ".." || startswith(relative, "../") || startswith(relative, "..\\")) && error("Anatomical asset path escapes benchmark/objects")
    bytes2hex(sha256(read(path))) == meta["sha256"] || error("Anatomical asset SHA-256 mismatch: $path")
    labels, spacing, center = h5open(path, "r") do file
        # HDF5.jl reverses multidimensional dataset axes relative to h5py.
        array = read(file["labels"])
        (permutedims(array, reverse(1:ndims(array))), read(file["voxel_size_m"]), read(file["center_m"]))
    end
    shape = size(labels)
    profile = ANATOMY["field_profiles"][meta["field_profile"]]
    properties = profile["properties"]
    isnothing(properties) && error("Field-specific tissue properties are not configured: $ref")
    count = sum(labels .!= 0)
    x = zeros(count); y = zeros(count); z = zeros(count)
    rho = zeros(count); t1 = zeros(count); t2 = zeros(count); delta = zeros(count)
    names = Vector{String}(undef, count)
    position = 0
    for index in CartesianIndices(labels)
        value = labels[index]
        value == 0 && continue
        position += 1
        for axis in 1:ndims(labels)
            (x,y,z)[axis][position] = center[axis] + (index[axis] - .5 - shape[axis]/2)*spacing[axis]
        end
        names[position] = ANATOMY["materials"][value]
        rho[position],t1[position],t2[position],delta[position] = properties[value]
    end
    obj = Phantom(name=ref, x=x, y=y, z=z, ρ=rho, T1=t1, T2=t2,
                  Δw=2π .* delta .* (field_strength_T/3))
    return obj, names
end

function material_at(ref, p)
    haskey(SHAPES["objects"], ref) || error("Unknown phantom: $ref")
    label = ""
    for shape in SHAPES["objects"][ref]
        n = length(shape["center_mm"])
        q = (p[1:n] .* 1000 .- shape["center_mm"]) ./ shape["radii_mm"]
        inside = shape["kind"] == "box" ? all(abs.(q) .<= 1) : sum(q.^2) <= 1
        inside && (label = shape["material"])
    end
    return label
end

"""Return a native Phantom and material labels, sampled at fixed voxel centers."""
function build_phantom(ref::String; field_strength_T=nothing)
    haskey(REGISTRY, ref) || error("Unknown phantom: $ref")
    haskey(ANATOMY["objects"], ref) && return anatomical_phantom(ref; field_strength_T)
    meta = REGISTRY[ref]
    axes = [((collect(0:n-1) .+ 0.5) ./ n .- 0.5) .* (fov/1000)
            for (n,fov) in zip(meta["grid"], meta["fov_mm"])]
    length(axes) == 2 && push!(axes, [0.0])
    xyz = NTuple{3,Float64}[]; labels = String[]
    for z in axes[3], y in axes[2], x in axes[1]
        label = material_at(ref, [x,y,z])
        isempty(label) && continue
        push!(xyz, (x,y,z)); push!(labels, label)
    end
    props = [MATERIALS[label] for label in labels]
    obj = Phantom(name=ref, x=getindex.(xyz,1), y=getindex.(xyz,2), z=getindex.(xyz,3),
                  ρ=getindex.(props,1), T1=getindex.(props,2), T2=getindex.(props,3))
    return obj, labels
end
