# Build context: clean KomaMRICore/ plus the three fixed adapter scripts.
# BASE is the PulseDesigner-enabled author image from the same source revision.
ARG BASE=mriseqbench/author-pulseq_julia:local
ARG GPU_JULIA_BASE=julia:1.12.6
FROM ${GPU_JULIA_BASE} AS gpu_julia
FROM ${BASE}
COPY --from=gpu_julia /usr/local/julia /usr/local/julia-gpu
ENV PATH=/opt/venv/bin:/usr/local/julia-gpu/bin:${PATH} JULIA_PATH=/usr/local/julia-gpu JULIA_VERSION=1.12.6
ENV JULIA_PKG_OFFLINE=false JULIA_LOAD_PATH=/opt/simulator:@stdlib JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_CUDA_HARD_MEMORY_LIMIT=2GiB
RUN /opt/venv/bin/pip install --no-cache-dir pypulseq==1.5.0.post1
COPY KomaMRICore /opt/sources/KomaMRICore
RUN python3 -c 'import pathlib,re; root=pathlib.Path("/opt/sources/KomaMRICore"); p=root/"Project.toml"; p.write_text(re.sub(r"(?ms)^\[workspace\]\n.*?(?=^\[|\Z)", "", p.read_text())); p=root/"src/KomaMRICore.jl"; p.write_text(p.read_text().replace("include(\"precompile.jl\")", "")); (root/"src/precompile.jl").unlink(missing_ok=True)'
# Use the native GPU evaluator's Julia/compiler stack; MRI source is unchanged.
# Prepare caches and lazy CUDA13.4 artifacts while online; services have no network.
RUN mkdir /opt/simulator && julia --startup-file=no --project=/opt/simulator -e 'using Pkg; Pkg.develop([PackageSpec(path="/opt/sources/KomaMRIBase"), PackageSpec(path="/opt/sources/KomaMRIFiles"), PackageSpec(path="/opt/sources/KomaMRICore")]); Pkg.add(["JSON", "HDF5"])' \
    && julia --startup-file=no --heap-size-hint=1G --project=/opt/simulator -e 'using Pkg; Pkg.add([PackageSpec(name="CUDA", version="6.4.0"), PackageSpec(name="CUDACore", version="6.4.0"), PackageSpec(name="CUDATools", version="6.4.0"), PackageSpec(name="GPUCompiler", version="2.10.0"), PackageSpec(name="GPUArrays", version="11.5.16"), PackageSpec(name="LLVM", version="9.13.2"), PackageSpec(name="LLVMExtra_jll", version="0.0.47"), PackageSpec(name="Atomix", version="1.2.1"), PackageSpec(name="UnsafeAtomics", version="0.3.2")]; preserve=Pkg.PRESERVE_ALL); using Base.BinaryPlatforms, TOML, UUIDs; config=TOML.parsefile("/opt/simulator/Project.toml"); extras=get!(config,"extras",Dict{String,Any}()); extras["CUDA_Runtime_jll"]="76a88914-d11a-5bdc-97e0-2f5a05c973a2"; extras["CUDA_Compiler_jll"]="d1e2174e-dfdc-576e-b43e-73b79eb1aca8"; open("/opt/simulator/Project.toml","w") do io; TOML.print(io,config); end; open("/opt/simulator/LocalPreferences.toml","w") do io; TOML.print(io,Dict("CUDA_Runtime_jll"=>Dict("version"=>"13.4"),"CUDA_Compiler_jll"=>Dict("version"=>"13.4"))); end; manifest=TOML.parsefile("/opt/simulator/Manifest.toml"); platform=HostPlatform(); platform["cuda"]="13.4"; for name in ["CUDA_Compiler_jll", "CUDA_Runtime_jll", "CUDA_Driver_jll", "NVPTX_LLVM_Backend_jll", "cuBLAS", "cuFFT", "cuRAND", "cuSOLVER", "cuSPARSE"]; file=Base.locate_package(Base.PkgId(UUID(manifest["deps"][name][1]["uuid"]),name)); artifact=joinpath(dirname(dirname(file)),"Artifacts.toml"); isfile(artifact) && Pkg.Artifacts.ensure_all_artifacts_installed(artifact; platform=platform,include_lazy=true); end; Pkg.precompile(); using CUDA; CUDA.precompile_runtime(); using KomaMRIBase, KomaMRIFiles, KomaMRICore, CUDA, JSON, HDF5' \
    && rm -rf /opt/julia-depot/registries /opt/julia-depot/logs
COPY simulate.py /opt/mriseqbench/simulate
COPY simulate_koma.jl /opt/mriseqbench/simulate_koma.jl
RUN chmod 755 /opt/mriseqbench/simulate
ENV JULIA_PKG_OFFLINE=true MRISEQBENCH_SIMULATOR=koma
CMD ["/opt/mriseqbench/simulate", "--probe"]
