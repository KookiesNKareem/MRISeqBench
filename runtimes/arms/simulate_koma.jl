using KomaMRIBase, KomaMRIFiles, KomaMRICore, JSON, HDF5
source, output = ARGS
case = JSON.parsefile(joinpath(source, "case.json"))
obj = read_phantom(joinpath(source, "phantom.phantom"))
seq = read_seq(joinpath(source, "sequence.seq"))
sys = Scanner(limits=HardwareLimits(B0=case["hardware"]["field_strength_T"]))
signal = simulate(obj, seq, sys;
                  sim_params=Dict{String,Any}("return_type"=>"mat", "gpu"=>false), verbose=false)
h5open(joinpath(output, "signal.h5"), "w") do f
    f["signal_real"] = real(signal)
    f["signal_imag"] = imag(signal)
end
open(joinpath(output, "simulation.json"), "w") do io
    JSON.print(io, Dict("engine"=>"koma", "diagnostic"=>true, "signal_shape"=>size(signal)))
end
