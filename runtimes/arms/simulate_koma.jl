using KomaMRIBase, KomaMRIFiles, KomaMRICore, CUDA, JSON, HDF5
using SimWarm
SimWarm.WARM || error("Simulator image was not prewarmed on a GPU")
CUDA.functional() || error("A functional CUDA GPU is required for simulation; no CPU fallback")
CUDA.allowscalar(false)
backend = KomaMRICore.get_backend(true)
backend isa KomaMRICore.KA.GPU || error("KomaMRI simulation cannot fall back to CPU")
source, output = ARGS
case = JSON.parsefile(joinpath(source, "case.json"))
obj = read_phantom(joinpath(source, "phantom.phantom"))
seq = read_seq(joinpath(source, "sequence.seq"))
sys = Scanner(limits=HardwareLimits(B0=case["hardware"]["field_strength_T"]))
signal = simulate(obj, seq, sys;
                  sim_params=Dict{String,Any}("return_type"=>"mat", "gpu"=>true), verbose=false)
h5open(joinpath(output, "signal.h5"), "w") do f
    f["signal_real"] = real(signal)
    f["signal_imag"] = imag(signal)
end
open(joinpath(output, "simulation.json"), "w") do io
    JSON.print(io, Dict("engine"=>"koma", "diagnostic"=>true, "signal_shape"=>size(signal),
                        "backend"=>"cuda", "device"=>CUDA.name(CUDA.device()), "cpu_fallback"=>false))
end
