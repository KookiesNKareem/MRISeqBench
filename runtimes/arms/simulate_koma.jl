using KomaMRIBase, KomaMRIFiles, KomaMRICore, JSON, HDF5
include(joinpath(@__DIR__, "reconstruct_koma.jl"))
const DEVICE = get(ENV, "MRISEQBENCH_SIMULATOR_DEVICE", "cuda")
DEVICE in ("cuda", "metal", "cpu") ||
    error("MRISEQBENCH_SIMULATOR_DEVICE must be 'cuda', 'metal', or 'cpu'")
if DEVICE == "cuda"
    @eval using CUDA
    @eval using SimWarm
    SimWarm.WARM || error("Simulator image was not prewarmed on a GPU")
    CUDA.functional() || error("A functional CUDA GPU is required; no CPU fallback")
    CUDA.allowscalar(false)
elseif DEVICE == "metal"
    Base.find_package("Metal") === nothing &&
        error("Metal backend requested, but Metal.jl is not installed in the selected Julia project")
    @eval using Metal
    Metal.functional() || error("A functional Metal GPU is required; no CPU fallback")
end
use_gpu = DEVICE != "cpu"
backend = KomaMRICore.get_backend(use_gpu; verbose=false)
expected = DEVICE == "cpu" ? "CPU" : DEVICE == "cuda" ? "CUDA" : "Metal"
KomaMRICore.name(backend) == expected ||
    error("Requested $DEVICE simulator, but KomaMRI selected $(KomaMRICore.name(backend))")
backend isa KomaMRICore.KA.GPU &&
    !(DEVICE == "cuda" ? occursin("CUDA", string(typeof(backend))) : occursin("Metal", string(typeof(backend)))) &&
    error("Requested $DEVICE simulator, but KomaMRI selected $(typeof(backend))")
DEVICE_NAME = DEVICE == "cpu" ? "CPU" : DEVICE == "cuda" ?
    CUDA.name(CUDA.device()) : KomaMRICore.device_name(backend)
function simulate_files(source, output)
    source, output = String(source), String(output)
    case = JSON.parsefile(joinpath(source, "case.json"))
    obj = read_phantom(joinpath(source, "phantom.phantom"))
    seq = read_seq(joinpath(source, "sequence.seq"))
    temporal = get(case["task"]["sequence"], "temporal", nothing)
    rr_ms = isnothing(temporal) ? nothing : get(temporal, "rr_interval_ms", nothing)
    rr_interval_s = isnothing(rr_ms) ? 1.0 : Float64(rr_ms) / 1000
    seq, physio = resolve_simulation_physio(seq, rr_interval_s)
    sys = Scanner(limits=HardwareLimits(B0=case["hardware"]["field_strength_T"]))
    signal = simulate(obj, seq, sys;
                      sim_params=Dict{String,Any}("return_type"=>"mat", "gpu"=>use_gpu), verbose=false)
    actual_backend = KomaMRICore.get_backend(use_gpu; verbose=false)
    KomaMRICore.name(actual_backend) == expected ||
        error("KomaMRI simulation changed backend during execution")
    h5open(joinpath(output, "signal.h5"), "w") do f
        f["signal_real"] = real(signal)
        f["signal_imag"] = imag(signal)
    end
    reconstruction = reconstruct_files(signal, seq, sys, output)
    open(joinpath(output, "simulation.json"), "w") do io
        JSON.print(io, Dict("engine"=>"koma", "diagnostic"=>true, "signal_shape"=>size(signal),
                            "backend"=>DEVICE, "device"=>DEVICE_NAME, "cpu_fallback"=>false,
                            "reconstruction"=>reconstruction, "physio"=>physio))
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("Usage: simulate_koma.jl INPUT_DIRECTORY OUTPUT_DIRECTORY")
    simulate_files(ARGS...)
end
