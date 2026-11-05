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
rr_ms = get(get(get(get(case, "task", Dict()), "sequence", Dict()), "temporal", Dict()), "rr_interval_ms", nothing)
if rr_ms !== nothing
    seq = resolve_triggers(seq, CardiacSignal(; heart_rate=1000.0 / Float64(rr_ms), first_peak=0.0))
end
sys = Scanner(limits=HardwareLimits(B0=case["hardware"]["field_strength_T"]))
has_adc = sum(adc.N for adc in seq.ADC; init=0) > 0
return_type = has_adc ? "mat" : "state"
result = simulate(obj, seq, sys;
                  sim_params=Dict{String,Any}("return_type"=>return_type, "gpu"=>true), verbose=false)
signal = has_adc ? result : result.xy
h5open(joinpath(output, "signal.h5"), "w") do f
    f["signal_real"] = real(signal)
    f["signal_imag"] = imag(signal)
end
open(joinpath(output, "simulation.json"), "w") do io
    JSON.print(io, Dict("engine"=>"koma", "diagnostic"=>true, "signal_shape"=>size(signal),
                        "backend"=>"cuda", "device"=>CUDA.name(CUDA.device()), "cpu_fallback"=>false))
end
