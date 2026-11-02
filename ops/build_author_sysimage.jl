using PackageCompiler, TOML, SHA, Libdl

length(ARGS) == 3 || error("Usage: julia --project=BUILDER ops/build_author_sysimage.jl AUTHOR_PROJECT OUTPUT_DIRECTORY ARM")
project, output, arm = abspath(ARGS[1]), abspath(ARGS[2]), ARGS[3]
arm in ("j0_koma_base", "j1_pulseq_julia", "j2_pulseq_julia_unitful",
        "j3_koma_base_koma", "j4_pulseq_julia_koma", "j5_pulseq_julia_unitful_koma",
        "j6_koma_base_julia_mcp", "j7_pulseq_julia_mcp_koma") || error("Unknown Julia arm: $arm")
pulse_designer = !(arm in ("j0_koma_base", "j3_koma_base_koma", "j6_koma_base_julia_mcp"))
unitful = arm in ("j2_pulseq_julia_unitful", "j5_pulseq_julia_unitful_koma")
source = read(joinpath(@__DIR__, "../src/mriseqbench/harness/arm_julia_session.jl"), String)
definitions = first(split(source, "\nMRISeqBenchJuliaSession.serve()"))
mkpath(output)
workload = joinpath(output, "generic-workload.jl")
write(workload, "Base.include_string(Main, " * repr(definitions) * ");\nMRISeqBenchJuliaSession.warmup($pulse_designer, $unitful)\nusing FFTW; fft(ones(ComplexF64, 8))\n")
image = joinpath(output, "author." * Libdl.dlext)
packages = [:KomaMRIBase, :KomaMRIFiles, :FFTW]
unitful && push!(packages, :Unitful)
started = time()
create_sysimage(packages; project, sysimage_path=image,
    precompile_execution_file=workload, cpu_target="native", incremental=true,
    instantiate=false)
metadata = Dict("arm"=>arm, "project"=>project, "julia"=>string(VERSION),
    "build_s"=>time()-started, "manifest_sha256"=>bytes2hex(sha256(read(joinpath(project,"Manifest.toml")))),
    "source_sha256"=>bytes2hex(sha256(source)), "sysimage_sha256"=>bytes2hex(sha256(read(image))))
open(joinpath(output, "build.toml"), "w") do io
    TOML.print(io, metadata)
end
