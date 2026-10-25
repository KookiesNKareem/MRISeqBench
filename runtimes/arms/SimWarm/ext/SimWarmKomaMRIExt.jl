module SimWarmKomaMRIExt
using KomaMRI, PrecompileTools, SimWarm

@compile_workload SimWarm.WARM && SimWarm.workload()
end
