module MRISeqBenchPhantoms
using KomaMRIBase, KomaMRIFiles, JSON, HDF5, SHA
export build_phantom, load_case, snapshot, materialize

include(joinpath(@__DIR__, "..", "..", "..", "benchmark", "objects", "phantoms.jl"))

struct PhantomCase
    phantom::Phantom{Float64}
    labels::Vector{String}
    effects::Vector
    source::Dict
end

include(joinpath(@__DIR__, "..", "..", "..", "benchmark", "physics", "effects.jl"))

function load_case(source::Dict)
    obj,labels=build_phantom(source["object"]["phantom"])
    effects=get(source["physics"],"effects",Any[])
    for e in effects
        kind=e["kind"]; model=e["model"]
        kind in ("b0","b1","motion","time_variation") || error("Unsupported effect kind: $kind")
        if kind in ("b0","b1")
            model in ("constant","linear","gaussian") || error("Unsupported field model: $model")
        elseif kind == "motion"
            model in ("rigid_sinusoidal","rigid_keyframes","affine_sinusoidal") || error("Unsupported motion model: $model")
        else
            model == "sinusoidal" || error("Unsupported time model: $model")
            e["target"] in ("b0_offset","b1_scale","proton_density") || error("Unsupported variation target")
            get(e,"target","") == "proton_density" && !(e["material"] in labels) && error("Unknown phantom material")
        end
    end
    count(e->e["kind"]=="motion",effects)<=1 || error("One motion effect is supported per profile")
    return PhantomCase(obj,labels,effects,source)
end
load_case(filename::String)=load_case(JSON.parsefile(filename))

function materialize(c::PhantomCase,out::String)
    names=("phantom.phantom","fields.h5","manifest.json")
    any(ispath(joinpath(out,f)) for f in names) && error("Refusing to overwrite phantom artifacts")
    mkpath(out)
    obj,b1=snapshot(c,0.0)
    native_motion!(c); obj.motion=c.phantom.motion
    # Keep native motion anchored to the untransformed geometry, including phase offsets.
    obj.x=copy(c.phantom.x); obj.y=copy(c.phantom.y); obj.z=copy(c.phantom.z)
    write_phantom(obj,joinpath(out,"phantom.phantom"))
    h5open(joinpath(out,"fields.h5"),"w") do f
        attributes(f)["schema_version"]=1
        attributes(f)["effects_json"]=JSON.json(c.effects)
        attributes(f)["frame"]="scanner"
        f["b1_scale_t0"]=b1
        f["material_labels"]=c.labels
    end
    needs_update=any(e->e["kind"]=="time_variation",c.effects) ||
        (any(e->e["kind"]=="motion",c.effects) && any(e->e["kind"] in ("b0","b1"),c.effects))
    manifest=Dict("schema_version"=>1,"phantom"=>c.source["object"]["phantom"],
        "seed"=>get(c.source,"seed",42),"spins"=>length(obj),"materials"=>unique(c.labels),
        "native_type"=>"KomaMRIBase.Phantom","koma_mri_base"=>string(pkgversion(KomaMRIBase)),
        "koma_mri_files"=>string(pkgversion(KomaMRIFiles)),"requires_field_updates"=>needs_update,
        "b1"=>"fields.h5:b1_scale_t0; apply per spin to RF in simulation",
        "snapshot_loader"=>"MRISeqBenchPhantoms.snapshot(load_case(case_json), time_s)",
        "motion_intervals_per_cycle"=>1024,
        "motion_interpolation"=>"linear native time curves (1024 intervals/cycle or keyframe segment); previous keyframes jump between adjacent Float64 instants",
        "files"=>Dict(f=>bytes2hex(sha256(read(joinpath(out,f)))) for f in names[1:2]))
    open(joinpath(out,"manifest.json"),"w") do io; JSON.print(io,manifest,2); write(io,'\n'); end
    return manifest
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS)==2 || error("Usage: loader.jl case.json output_directory")
    MRISeqBenchPhantoms.materialize(MRISeqBenchPhantoms.load_case(ARGS[1]),ARGS[2])
end
