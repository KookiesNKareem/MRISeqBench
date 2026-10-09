const REGISTRY = JSON.parsefile(joinpath(@__DIR__, "registry.json"))
const MATERIALS = Dict(
    "water" => (1.0, 1.0, 0.1), "wm" => (0.7, 0.8, 0.08),
    "gm" => (0.8, 1.3, 0.1), "csf" => (1.0, 4.0, 2.0),
    "fat" => (0.9, 0.25, 0.06), "background" => (0.8, 1.2, 0.09),
    "inclusion" => (1.0, 0.7, 0.06))

function material_at(ref, p)
    x, y, z = p .* 1000
    if ref == "uniform_disc@1"
        return x^2 + y^2 <= 60^2 ? "water" : ""
    elseif ref == "tissue_discs@1"
        for (name, cx, cy, r) in (("wm",-35,30,24), ("gm",35,30,24),
                                  ("csf",0,-40,23), ("fat",-45,-45,12))
            (x-cx)^2 + (y-cy)^2 <= r^2 && return name
        end
        return ""
    elseif ref == "nested_ellipses@1"
        (x/75)^2 + (y/60)^2 > 1 && return ""
        return ((x-25)/17.5)^2 + ((y-10)/25)^2 <= 1 ? "inclusion" : "background"
    elseif ref == "volume_ellipsoids@1"
        # Inclusion overwrites background, matching ordered shape semantics.
        abs(x-20) <= 15 && abs(y) <= 12.5 && abs(z) <= 10 && return "inclusion"
        return (x/75)^2 + (y/60)^2 + (z/40)^2 <= 1 ? "background" : ""
    end
    error("Unknown phantom: $ref")
end

"""Return a native Phantom and material labels, sampled at fixed voxel centers."""
function build_phantom(ref::String)
    haskey(REGISTRY, ref) || error("Unknown phantom: $ref")
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
