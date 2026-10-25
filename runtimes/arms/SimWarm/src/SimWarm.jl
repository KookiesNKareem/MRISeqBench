"""Caches the simulator adapters' CUDA compilation in pkgimages.

Precompiled on a GPU by `prewarm_simulator.py`; adapters require `WARM` so no
simulation call pays the ~1 minute first-run compilation against agent time.
`SimWarmKomaMRIExt` repeats the workload with KomaMRI loaded (komamripy), whose
method additions otherwise invalidate this cache.
"""
module SimWarm
using KomaMRIBase, KomaMRIFiles, KomaMRICore, CUDA, PrecompileTools

# True only in a cache compiled on a GPU; loading from source (no cache) is cold.
const WARM = ccall(:jl_generating_output, Cint, ()) == 1 && CUDA.functional()

"""The adapters' file-based calls; keyword sets select specializations."""
function workload()
    mktempdir() do dir
        sys = Scanner(limits=HardwareLimits(B0=3.0))
        rf = PulseDesigner.make_block_pulse(π / 6; duration=1e-3, sys, use=Excitation())
        adc = PulseDesigner.make_adc(32; duration=3.2e-3, sys)
        seq = Sequence(sys)
        addblock!(seq, rf, Duration(dur(rf, sys)))
        addblock!(seq, adc, Duration(dur(adc, sys)))
        write_seq(seq, joinpath(dir, "s.seq"); sys, verbose=false)
        write_phantom(brain_phantom2D(), joinpath(dir, "p.phantom"))
        obj = read_phantom(joinpath(dir, "p.phantom"))
        params = Dict{String,Any}("return_type" => "mat", "gpu" => true)
        for seq in (read_seq(joinpath(dir, "s.seq")), PulseDesigner.EPI_example())
            simulate(obj, seq, sys; sim_params=params, verbose=false)  # simulate_koma.jl
            simulate(obj, seq, sys; sim_params=params)  # komamripy's call
        end
    end
end

@compile_workload WARM && workload()
end
