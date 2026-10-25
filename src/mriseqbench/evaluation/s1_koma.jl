# Headless KomaMRI engine. The only integration kernels below are KomaMRICore's.
using KomaMRICore, KomaMRIFiles, HDF5, JSON, LinearAlgebra, Logging
include(joinpath(@__DIR__, "..", "phantoms", "loader.jl"))
using .MRISeqBenchPhantoms
const CoreMRI = KomaMRICore
const SEQUENCES = Dict{String,Any}()

function read_pulseq_sequence(path)
    # Parse failures of the submitted bytes are tagged so the judge can fail
    # the candidate; host failures keep their own exception types.
    seq = try
        read_seq(path)
    catch e
        e isa Union{OutOfMemoryError,InterruptException,SystemError,Base.IOError} && rethrow()
        error("Native Pulseq reader rejected the file: " * sprint(showerror, e))
    end
    # Files 0.12.0 decodes ADC(1) with delay=PulseqDelay+dwell/2 and
    # T=dwell. Base 0.15.1's times(ADC(1)) adds T/2 once more. Remove
    # only that duplicate half-dwell; keep the actual Pulseq sample centre.
    # Multi-sample ADCs and the submitted file bytes are unchanged. The strict
    # Python/native ADC timestamp check below remains the independent guard.
    for adc in seq.ADC
        adc.N == 1 && (adc.delay -= adc.T / 2)
    end
    return seq
end

const DEVICE = get(ENV, "MRISEQBENCH_KOMA_DEVICE", "cuda")
DEVICE == "cuda" || error("MRISeqBench simulations require CUDA; CPU execution is disabled")
using CUDA
CUDA.functional() || error("A functional CUDA GPU is required for grading; no CPU fallback")
CUDA.allowscalar(false)

function simulation_backend()
    backend = CoreMRI.get_backend(true)
    backend isa CoreMRI.KA.GPU || error("CUDA grading cannot fall back to CPU")
    CoreMRI.KA.supports_float64(backend) || error("Grading requires Float64 on the GPU")
    return backend
end

on_device(x) = CoreMRI.gpu(x)

function update_device_fields!(obj, host_obj, c, mid, offsets)
    case_field_update!(host_obj,c,mid,offsets)
    copyto!(obj.ρ,host_obj.ρ)
    copyto!(obj.Δw,host_obj.Δw)
end

function execution_device()
    Dict("backend"=>DEVICE, "precision"=>"f64",
         "device"=>CUDA.name(CUDA.device()),
         "cuda_version"=>string(pkgversion(CUDA)),
         "cpu_fallback"=>false)
end

function case_field_update!(obj, c, t, offsets)
    p = hcat(c.phantom.x, c.phantom.y, c.phantom.z)
    for e in c.effects
        e["kind"] == "motion" || continue
        p = MRISeqBenchPhantoms.coordinates(c.phantom, e, t)
    end
    b0 = copy(offsets)
    obj.ρ .= c.phantom.ρ
    for e in c.effects
        if e["kind"] == "b0"
            b0 .+= MRISeqBenchPhantoms.spatial_field(e,p)
        elseif e["kind"] == "time_variation"
            s = MRISeqBenchPhantoms.sinusoid(e,t)
            if e["target"] == "b0_offset"
                b0 .+= e["amplitude_hz"]*s
            elseif e["target"] == "proton_density"
                obj.ρ[c.labels .== e["material"]] .*= 1 + e["amplitude_scale"]*s
            else
                error("Unsupported time variation target")
            end
        elseif e["kind"] != "motion"
            error("Unsupported effect kind: $(e["kind"])")
        end
    end
    obj.Δw .= 2π .* b0
    return nothing
end

# Use the current upstream sampler; RF samples/support and explicit waveform
# boundaries are retained by its public sampling rule rather than historical
# fixes for KomaMRIBase 0.11's interpolation grid.
struct CaseTimeSampling{R} <: CoreMRI.KomaMRIBase.SamplingRule
    rule::R
    maxdt::Float64
end
CoreMRI.KomaMRIBase.preserved_samples(r::CaseTimeSampling) = CoreMRI.KomaMRIBase.preserved_samples(r.rule)
function CoreMRI.KomaMRIBase.additional_sampling_times(r::CaseTimeSampling, times, context::CoreMRI.KomaMRIBase.BlockSamplingContext)
    extra = CoreMRI.KomaMRIBase.additional_sampling_times(r.rule, times, context)
    n = max(1,ceil(Int,context.duration/r.maxdt))
    return sort!(unique!(vcat(extra,collect(range(0.0,context.duration;length=n+1)))))
end

# The native discretized timeline must be monotone, with nonnegative steps,
# inside [0, duration]; otherwise the file means something else natively.
function check_native_timeline(seqd, seq)
    t, T = seqd.t, dur(seq)
    # Block-boundary rounding can leave steps of about -1e-17 s.
    ok = all(isfinite, t) && all(>=(-1e-12), seqd.Δt) && all(>=(-1e-12), diff(t)) &&
        first(t) >= -1e-9 && last(t) <= T + 1e-9
    ok || error("Native consistency: discretized timeline outside a monotone [0, $T] s " *
                "(min $(minimum(t)) s, max $(maximum(t)) s)")
    # RF (amplitude and RF-frame phase), frequency and gradient samples must be
    # finite before GPU dispatch; finite file tokens can still overflow natively.
    for (name, values) in (("B1", seqd.B1), ("Δf", seqd.Δf), ("Gx", seqd.Gx),
                           ("Gy", seqd.Gy), ("Gz", seqd.Gz))
        all(isfinite, values) || error("Native consistency: non-finite native $name samples")
    end
    return seqd
end

function discretize_case(seq, params, motion, has_dynamic)
    rule = CoreMRI.simulation_sampling_rule(params["sim_method"],params)
    if has_dynamic
        # Add the dynamic-field time grid before the method inserts its RF
        # quadrature nodes, so higher-order integration groups stay intact.
        rule = rule isa CoreMRI.IntegrationNodeSamplingRule ?
            CoreMRI.IntegrationNodeSamplingRule(rule.method,CaseTimeSampling(rule.rule,params["Δt"])) :
            CaseTimeSampling(rule,params["Δt"])
    end
    return check_native_timeline(discretize(seq; sampling_rule=rule, motion,
                                            freq_in_phase=params["freq_in_phase"]), seq)
end

has_dynamic_fields(c) = any(e->e["kind"]=="time_variation",c.effects) ||
    (any(e->e["kind"]=="motion",c.effects) && any(e->e["kind"]=="b0",c.effects))

function simulate_case(obj, seq, sys, c, offsets, params; individual=false)
    has_dynamic = has_dynamic_fields(c)
    case_field_update!(obj,c,0.0,offsets)
    # Time-dependent fields/equilibrium are updated at each integration midpoint.
    # State persists throughout; every step delegates rotation/relaxation/acquisition
    # to the pinned KomaMRI kernels, including their native motion coordinates.
    method = individual ? CoreMRI.BlochDict() : params["sim_method"]
    fuse_rf = get(params,"fuse_diagnostic_rf",false) && individual &&
        !has_dynamic && obj.motion isa NoMotion
    # Static S4 replay can use the public simulator end to end. The file already
    # contains upstream-resolved waits; NoPhysioSignal prevents double waiting.
    if !has_dynamic && get(params,"spoil_times_s",nothing) === nothing && !fuse_rf
        simulation_backend() # Fail closed before upstream's automatic fallback.
        discretize_case(seq,params,obj.motion,false) # Timeline check only.
        p = copy(params)
        p["sim_method"] = method
        p["max_block_length"] = 256
        p["max_rf_block_length"] = 256
        try
            result = CoreMRI.simulate(obj,seq,sys;sim_params=p,verbose=false,physio=NoPhysioSignal())
            return individual ? dropdims(result;dims=4) : result
        catch err
            # Upstream's own discretisation can leave an RF block with an
            # interval count that is not a multiple of the Magnus step (e.g. 61
            # for a 2-sample block pulse) and rejects the file instead of
            # simulating it. The explicit replay below discretises with the
            # method's integration nodes and otherwise runs the same pinned
            # kernels; take it for that file rather than fail the submission.
            occursin("not a multiple of", sprint(showerror, err)) || rethrow()
        end
    end
    seqd = discretize_case(seq,params,obj.motion,has_dynamic)
    stride = CoreMRI.eval_intervals_per_step(method)
    if stride > 1
        # Two event times inside a pulse can differ by rounding noise only, e.g.
        # the RF centre against a gradient raster point. Upstream flags that
        # degenerate interval as excitation, but its interior quadrature nodes
        # collapse onto its endpoints, so the RF run is one interval short of a
        # whole integration step and the public simulator rejects the file.
        # A sub-picosecond interval carries no rotation: integrate it as free
        # precession so the remaining RF intervals form whole steps again.
        degenerate = seqd.excitation_bool .& (seqd.Δt .<= 1e-12)
        seqd.excitation_bool[degenerate] .= false
    end
    spoil_times = get(params,"spoil_times_s",nothing)
    # Bound GPU temporaries, including the per-spin diagnostic dictionaries.
    # The discretization and every ADC/integration step remain unchanged.
    block_length = has_dynamic ? 1 : 256
    ranges, rf_on = CoreMRI.get_sim_ranges(seqd;max_block_length=block_length,max_rf_block_length=block_length,eval_intervals_per_step=CoreMRI.eval_intervals_per_step(method))

    if spoil_times !== nothing && !has_dynamic
        # Retain the exact one-interval implementation's reset locations while
        # batching native GPU work between resets. Never batch dynamic fields.
        rf_steps = findall(seqd.excitation_bool)
        end_times = seqd.t[rf_steps .+ 1]
        reset_steps = sort!(unique!([rf_steps[j] for t in spoil_times
            for j in (searchsortedlast(end_times,t)+1,)
            if j <= length(rf_steps)]))
        split_ranges, split_rf = UnitRange{Int}[], Bool[]
        for (r,rf) in zip(ranges,rf_on)
            cuts = rf ? [first(r); filter(i->first(r)<i<last(r),reset_steps); last(r)] : [first(r); last(r)]
            for j in 1:(length(cuts)-1)
                push!(split_ranges,cuts[j]:cuts[j+1]);push!(split_rf,rf)
            end
        end
        ranges,rf_on = split_ranges,split_rf
    end
    state, obj = CoreMRI.initialize_spins_state(obj,method)
    state.z = copy(state.z) # Keep evolving Mz separate from equilibrium rho.
    backend = simulation_backend()
    host_obj, host_seqd = obj, seqd
    obj, seqd, state = on_device(obj), on_device(seqd), on_device(state)
    groupsize = CoreMRI.DEFAULT_PRECESSION_GROUPSIZE
    max_adc = maximum(r -> sum(@view host_seqd.ADC[(first(r)+1):last(r)]), ranges; init=0)
    prealloc = CoreMRI.prealloc(method,backend,obj,state,maximum(length.(ranges)),max_adc,groupsize,sys)
    signal = on_device(individual ? zeros(ComplexF64,sum(seq.ADC.N),length(obj),1,1) : zeros(ComplexF64,sum(seq.ADC.N),1,1))
    rf_method = CoreMRI.Bloch()
    rf_prealloc = fuse_rf ? CoreMRI.prealloc(rf_method,backend,obj,state,maximum(length.(ranges)),0,groupsize,sys) : nothing
    rf_signal = fuse_rf ? on_device(zeros(ComplexF64,0,1,1)) : nothing
    sample = 1
    next_spoil = 1
    for (range,rf) in zip(ranges,rf_on)
        host_step = @view host_seqd[range]
        step = @view seqd[range]
        mid = (first(host_step.t)+last(host_step.t))/2
        count = sum(host_step.ADC[2:end])
        if spoil_times !== nothing && rf
            # Destroy old transverse coherence before each excitation while
            # retaining longitudinal saturation and recovery. Composite RF
            # subpulses share one imported RF event and one reset.
            while next_spoil <= length(spoil_times) && spoil_times[next_spoil] < last(host_step.t)
                fill!(state.xy,0)
                next_spoil += 1
            end
        end
        has_dynamic && update_device_fields!(obj,host_obj,c,mid,offsets)
        hasproperty(prealloc,:ΔBz) && (prealloc.ΔBz .= obj.Δw ./ (2π * γ))
        slots = sample:(sample+count-1)
        if rf && fuse_rf && count == 0
            # Same discretized left-endpoint rotations/relaxation and shared state;
            # keep dictionary acquisition whenever RF and ADC overlap.
            CoreMRI.run_spin_excitation_parallel!(obj,step,rf_signal,state,sys,rf_method,groupsize,backend,rf_prealloc;Nthreads=1)
        elseif rf
            target = individual ? @view(signal[slots,:,:,:]) : @view(signal[slots,:,:])
            CoreMRI.run_spin_excitation_parallel!(obj,step,target,state,sys,method,groupsize,backend,prealloc;Nthreads=1)
        else
            target = individual ? @view(signal[slots,:,:,:]) : @view(signal[slots,:,:])
            CoreMRI.run_spin_precession_parallel!(obj,step,target,state,sys,method,groupsize,backend,prealloc;Nthreads=1)
        end
        CoreMRI.KA.synchronize(backend)
        sample += count
    end
    return CoreMRI.cpu(individual ? dropdims(signal;dims=4) : sum(signal;dims=3))
end

# Receive encoding accumulates each short ADC block from native per-spin Bloch
# responses. State is advanced once, independent of receive sensitivity. This
# is exactly the same density-linear operation as separate coil-weighted Bloch
# replays, with bounded memory instead of a full ADC-by-spins dictionary.
# Exact static per-spin receive maps, materialized once by the native GPU
# receiver API. No interpolation or change to sensitivities/physical dynamics.
struct ExactStaticReceive{M} <: CoreMRI.KomaMRIBase.AbstractRFReceiveSystem
    weights::M
end
CoreMRI.KomaMRIBase.get_n_coils(r::ExactStaticReceive) = size(r.weights,2)
CoreMRI.prealloc_receiver(r::ExactStaticReceive, obj, backend, ::NoMotion) = CoreMRI.gpu(r.weights,backend)

function simulate_receive_case(obj,seq,sys,c,offsets,params,receive)
    has_dynamic = has_dynamic_fields(c)
    case_field_update!(obj,c,0.0,offsets)
    if !has_dynamic && obj.motion isa NoMotion
        size(receive,1)==length(obj) || error("Receive/spin dimension mismatch")
        simulation_backend() # Fail closed before native automatic fallback.
        scanner = Scanner(limits=sys.limits, gradient=sys.gradient,
                          receiver=ExactStaticReceive(receive), transmitter=sys.transmitter)
        p = copy(params)
        p["sim_method"] = CoreMRI.Bloch()
        p["max_block_length"] = 256
        p["max_rf_block_length"] = 256
        # Native fused GPU acquisition accumulates the same linear coil sums as
        # BlochDict followed by matrix multiplication, without per-step broadcasts.
        result = CoreMRI.simulate(obj,seq,scanner;sim_params=p,verbose=false,physio=NoPhysioSignal())
        return reshape(result,sum(seq.ADC.N),size(receive,2))
    end
    method = CoreMRI.BlochDict()
    seqd = discretize_case(seq,params,obj.motion,has_dynamic)
    block_length = has_dynamic ? 1 : 256
    ranges, rf_on = CoreMRI.get_sim_ranges(seqd;max_block_length=block_length,max_rf_block_length=block_length,eval_intervals_per_step=CoreMRI.eval_intervals_per_step(method))
    state, obj = CoreMRI.initialize_spins_state(obj,method)
    state.z = copy(state.z)
    backend = simulation_backend()
    host_obj, host_seqd = obj, seqd
    obj, seqd, state = on_device(obj), on_device(seqd), on_device(state)
    groupsize = CoreMRI.DEFAULT_PRECESSION_GROUPSIZE
    max_adc = maximum(r -> sum(@view host_seqd.ADC[(first(r)+1):last(r)]), ranges; init=0)
    prealloc = CoreMRI.prealloc(method,backend,obj,state,maximum(length.(ranges)),max_adc,groupsize,sys)
    size(receive,1)==length(obj) || error("Receive/spin dimension mismatch")
    signal = on_device(zeros(ComplexF64,sum(seq.ADC.N),size(receive,2)))
    receive = on_device(receive)
    sample = 1
    for (range,rf) in zip(ranges,rf_on)
        host_step = @view host_seqd[range]
        step = @view seqd[range]
        has_dynamic && update_device_fields!(obj,host_obj,c,(first(host_step.t)+last(host_step.t))/2,offsets)
        count = sum(host_step.ADC[2:end])
        target = on_device(zeros(ComplexF64,count,length(obj),1,1))
        if rf
            CoreMRI.run_spin_excitation_parallel!(obj,step,target,state,sys,method,groupsize,backend,prealloc;Nthreads=1)
        else
            CoreMRI.run_spin_precession_parallel!(obj,step,target,state,sys,method,groupsize,backend,prealloc;Nthreads=1)
        end
        if count > 0
            @views mul!(signal[sample:sample+count-1,:],reshape(target,count,length(obj)),receive)
        end
        CoreMRI.KA.synchronize(backend)
        sample += count
    end
    return CoreMRI.cpu(signal)
end

function simulate_request(req)
    seq = deepcopy(get!(SEQUENCES,req["sequence_sha256"]) do
        read_pulseq_sequence(req["sequence"])
    end)
    # Receiver demodulation uses the exact original Pulseq phases in the judge;
    # prevent duplicate demodulation in KomaMRI, including diagnostic acquisitions.
    for adc in seq.ADC
        adc.ϕ = 0.0
        adc.Δf = 0.0
    end
    active_rf = [rf for rf in seq.RF if !iszero(sum(abs.(rf.A)))]
    length(active_rf) == length(req["rf_frequency_hz"]) || error("Native consistency: KomaMRI/Pulseq RF count mismatch")
    for (i,rf) in enumerate(active_rf)
        rf.Δf = req["rf_frequency_hz"][i]
        rf.ϕ += req["rf_phase_correction_rad"][i]
    end
    input = h5open(req["input"],"r") do f
        Dict(k=>read(f[k]) for k in keys(f))
    end
    times = Float64.(get_adc_sampling_times(seq))
    length(times)==length(input["adc_times"]) || error("Native consistency: KomaMRI/Pulseq ADC count mismatch")
    maximum(abs.(times-input["adc_times"])) <= 1e-9 || error("Native consistency: KomaMRI/Pulseq ADC time mismatch: $(maximum(abs.(times-input["adc_times"]))) s")
    obj = Phantom(name="S1 registered-shape quadrature",x=input["x"],y=input["y"],z=input["z"],
                  ρ=input["rho"],T1=input["t1"],T2=input["t2"])
    labels = [string(req["materials"][i+1]) for i in input["labels"]]
    c = MRISeqBenchPhantoms.PhantomCase(deepcopy(obj),labels,req["effects"],Dict())
    MRISeqBenchPhantoms.native_motion!(c)
    obj.motion = c.phantom.motion
    c.phantom.motion = NoMotion()
    integrator = get(req,"integration_method","bloch")
    integrator in ("bloch","magnus4") || error("Unsupported integration method: $integrator")
    params = CoreMRI.default_sim_params(Dict{String,Any}(
        "gpu"=>true,"precision"=>"f64","return_type"=>"mat",
        "sim_method"=>(integrator=="magnus4" ? CoreMRI.BlochMagnus4() : CoreMRI.Bloch()),
        "Δt"=>req["step_s"],"Δt_rf"=>req["rf_step_s"],
        "max_block_length"=>Inf,"max_rf_block_length"=>Inf,
        "preserve_samples"=>(:rf,:gradients,:rf_center),
        "Nthreads"=>min(Threads.nthreads(),length(obj))))
    params["fuse_diagnostic_rf"] = get(req,"fuse_diagnostic_rf",true)
    individual = get(req,"individual",false)
    params["spoil_times_s"] = get(req,"spoil_times_s",nothing)
    if individual
        params["sim_method"] = CoreMRI.BlochDict()
        params["Nthreads"] = 1
    end
    if req["demodulate_encoding"]
        isempty(req["effects"]) || error("Encoding-demodulated probes require ideal physics")
        params["sim_method"] = CoreMRI.BlochDict()
        obj.Δw .= 2π .* input["offset_hz"]
        individual = simulate_case(obj,seq,Scanner(limits=HardwareLimits(B0=req["field_strength_T"])),c,input["offset_hz"],params;individual=true)
        signal = zeros(ComplexF64,length(times))
        for i in eachindex(times), s in eachindex(obj.x)
            phase = 2π * (input["k0"][i]*obj.x[s] + input["k1"][i]*obj.y[s] + input["k2"][i]*obj.z[s])
            signal[i] += individual[i,s,1] * cis(phase)
        end
    elseif haskey(input,"receive_weights")
        signal = simulate_receive_case(obj,seq,Scanner(limits=HardwareLimits(B0=req["field_strength_T"])),c,input["offset_hz"],params,input["receive_weights"])
    else
        output = simulate_case(obj,seq,Scanner(limits=HardwareLimits(B0=req["field_strength_T"])),c,input["offset_hz"],params;individual)
        signal = individual ? output[:,:,1] : vec(output)
    end
    h5open(req["output"],"w") do f
        f["signal_real"] = real(signal)
        f["signal_imag"] = imag(signal)
    end
end

# Independent unitary inversion assay. One CUDA thread per detuning retains every
# original RF interval, with no relaxation, encoding or CPU dynamics fallback.
function inversion_assay_kernel!(inverted, offsets, real_rf, imag_rf, widths)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    if index <= length(offsets)
        mx, my, mz = 0.0, 0.0, 1.0
        for j in eachindex(widths)
            scale = 2π * widths[j]
            ax, ay, az = scale * real_rf[j], scale * imag_rf[j], scale * offsets[index]
            angle = sqrt(ax * ax + ay * ay + az * az)
            denominator = max(angle, 1e-30)
            ux, uy, uz = ax / denominator, ay / denominator, az / denominator
            sine, cosine = sincos(angle)
            dot = ux * mx + uy * my + uz * mz
            nx = mx * cosine - (uy * mz - uz * my) * sine + ux * dot * (1 - cosine)
            ny = my * cosine - (uz * mx - ux * mz) * sine + uy * dot * (1 - cosine)
            nz = mz * cosine - (ux * my - uy * mx) * sine + uz * dot * (1 - cosine)
            mx, my, mz = nx, ny, nz
        end
        inverted[index] = -mz
    end
    return nothing
end

function inversion_assay_request(req)
    simulation_backend()
    input = h5open(req["input"], "r") do f
        Dict(k => read(f[k]) for k in keys(f))
    end
    offsets = CUDA.CuArray(Float64.(input["offset_hz"]))
    real_rf = CUDA.CuArray(Float64.(input["rf_real_hz"]))
    imag_rf = CUDA.CuArray(Float64.(input["rf_imag_hz"]))
    widths = CUDA.CuArray(Float64.(input["width_s"]))
    inverted = similar(offsets)
    CUDA.@cuda threads=32 blocks=cld(length(offsets), 32) inversion_assay_kernel!(inverted, offsets, real_rf, imag_rf, widths)
    CUDA.synchronize()
    h5open(req["output"], "w") do f
        f["inverted_mz"] = Array(inverted)
    end
end

function resolve_physio_request(req)
    # Resolve with the upstream scheduler before Python inspection, SAR, target
    # generation or native replay. The returned waits become explicit delays;
    # the custom kernel bridge must not resolve them for a second time.
    seq = read_pulseq_sequence(req["sequence"])
    signal = CardiacSignal(; heart_rate=Float64(req["heart_rate_hz"]),
                            first_peak=Float64(req["first_peak_s"]))
    resolved = resolve_triggers(seq, signal)
    before, after = Float64.(seq.DUR), Float64.(resolved.DUR)
    Dict("ok"=>true, "block_durations_before_s"=>before,
         "block_durations_after_s"=>after, "waits_s"=>after .- before,
         "adc_times_s"=>Float64.(get_adc_sampling_times(resolved)),
         "resolver"=>"KomaMRIBase.resolve_triggers/CardiacSignal",
         "versions"=>Dict("KomaMRIBase"=>string(pkgversion(KomaMRIBase)),
                         "KomaMRICore"=>string(pkgversion(KomaMRICore)),
                         "KomaMRIFiles"=>string(pkgversion(KomaMRIFiles))))
end

for line in eachline(stdin)
    response = try
        req = JSON.parse(line)
        if req["operation"] == "versions"
            Dict("ok"=>true,"versions"=>Dict("KomaMRICore"=>string(pkgversion(KomaMRICore)),
                "KomaMRIBase"=>string(pkgversion(KomaMRIBase)),"KomaMRIFiles"=>string(pkgversion(KomaMRIFiles)),
                "Julia"=>string(VERSION)),"execution"=>execution_device())
        elseif req["operation"] == "resolve_physio"
            redirect_stdout(stderr) do
                with_logger(NullLogger()) do
                    resolve_physio_request(req)
                end
            end
        else
            redirect_stdout(stderr) do
                with_logger(NullLogger()) do
                    if req["operation"] == "simulate"
                        simulate_request(req)
                    elseif req["operation"] == "inversion_assay"
                        inversion_assay_request(req)
                    else
                        error("Unsupported KomaMRI worker operation; reconstruction requires BART")
                    end
                end
            end
            Dict("ok"=>true,"execution"=>execution_device())
        end
    catch e
        Dict("ok"=>false,"error"=>sprint(showerror,e,catch_backtrace()))
    end
    println(JSON.json(response)); flush(stdout)
end
