# Headless KomaMRI engine. The only integration kernels below are KomaMRICore's.
using KomaMRICore, KomaMRIFiles, HDF5, JSON, LinearAlgebra, Logging
include(joinpath(@__DIR__, "..", "phantoms", "loader.jl"))
using .MRISeqBenchPhantoms
const CoreMRI = KomaMRICore
const SEQUENCES = Dict{String,Any}()

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

function clear_rf_outside_support!(seqd, seq)
    # KomaMRIBase 0.11.3 interpolates RF edges separated by 1e-14 s. At
    # accumulated block times, roundoff can leave a nonzero endpoint which is
    # then integrated across a long RF-free delay. Step halving cannot remove
    # this spurious tail. Enforce the imported pulse's exact open support using
    # the same block/start/end arithmetic as native get_variable_times.
    starts, stops = Float64[], Float64[]
    block_starts = get_block_start_times(seq)
    for (i,rf) in enumerate(seq.RF)
        iszero(sum(abs.(rf.A))) && continue
        start = block_starts[i] + rf.delay
        push!(starts,start)
        push!(stops,start + sum(rf.T))
    end
    pulse = 1
    for (i,t) in enumerate(seqd.t)
        while pulse <= length(stops) && t >= stops[pulse]
            pulse += 1
        end
        if pulse > length(stops) || t <= starts[pulse]
            seqd.B1[i] = 0
        end
    end
    return seqd
end

function preserve_rf_samples(seq, seqd)
    # Keep native pulse support and every imported RF envelope transition.
    # times(rf,:A) includes outer zero knots: waveform sample i is knot i+1.
    # Losing that offset drops a full coarse RF step after a plateau jump.
    base = CoreMRI.KomaMRIBase
    extra = Float64[]
    block_starts = get_block_start_times(seq)
    for (i,rf) in enumerate(seq.RF)
        iszero(sum(abs.(rf.A))) && continue
        start = block_starts[i] + rf.delay
        stop = start + sum(rf.T)
        append!(extra,[start,nextfloat(start),prevfloat(stop),stop])
        any(a->a != first(rf.A),rf.A) || continue
        knots = base.times(rf,:A)
        changes = findall(diff(rf.A) .!= 0)
        keep = sort!(unique!(vcat(1,length(knots),2,length(knots)-1,
                                  changes .+ 1,changes .+ 2)))
        append!(extra,block_starts[i] .+ knots[keep])
        append!(extra,block_starts[i] .+ base.times(rf,:Δf))
    end
    isempty(extra) && return seqd
    t = sort!(unique!(vcat(seqd.t,extra)))
    B1, Δf, ψ = base.get_rfs(seq,t)
    Gx, Gy, Gz = base.get_grads(seq,t)
    adc_times = Set(get_adc_sampling_times(seq))
    adc = [time in adc_times for time in t]
    return CoreMRI.DiscreteSequence(Gx,Gy,Gz,complex.(B1),Δf,ψ,adc,t,diff(t))
end

function retain_keyframe_jumps(seq, seqd, effects)
    # KomaMRIBase 0.11.3 adds native motion period endpoints, but omits
    # interior TimeCurve knots. A previous-pose jump must not be integrated
    # as a ramp across a gradient/RF step. Preserve both sides of the jump.
    extra = Float64[]
    for e in effects
        e["kind"] == "motion" && e["model"] == "rigid_keyframes" || continue
        get(e,"interpolation","previous") == "previous" || continue
        for pose in e["poses"][2:end]
            t = Float64(pose["time_s"])
            first(seqd.t) < t < last(seqd.t) || continue
            append!(extra,[prevfloat(t),t])
        end
    end
    isempty(extra) && return seqd
    t = sort!(unique!(vcat(seqd.t,extra)))
    base = CoreMRI.KomaMRIBase
    B1, Δf, ψ = base.get_rfs(seq,t)
    Gx, Gy, Gz = base.get_grads(seq,t)
    adc_times = Set(get_adc_sampling_times(seq))
    adc = [time in adc_times for time in t]
    return CoreMRI.DiscreteSequence(Gx,Gy,Gz,complex.(B1),Δf,ψ,adc,t,diff(t))
end

function discretize_case(seq, params, motion, has_dynamic, effects=Any[])
    seqd = preserve_rf_samples(seq,discretize(seq; sampling_params=params,motion))
    seqd = retain_keyframe_jumps(seq,seqd,effects)
    if has_dynamic
        # Native event sampling leaves RF/gradient-free delays unsplit. Fields
        # and equilibrium can still vary there, so temporal refinement must
        # also bound those intervals. Retain every original event/ADC knot.
        t = copy(seqd.t)
        for i in eachindex(seqd.Δt)
            count = ceil(Int,seqd.Δt[i]/params["Δt"])
            count > 1 && append!(t,range(seqd.t[i],seqd.t[i+1];length=count+1))
        end
        sort!(unique!(t))
        base = CoreMRI.KomaMRIBase
        B1, Δf, ψ = base.get_rfs(seq,t)
        Gx, Gy, Gz = base.get_grads(seq,t)
        adc_times = Set(get_adc_sampling_times(seq))
        adc = [time in adc_times for time in t]
        seqd = CoreMRI.DiscreteSequence(Gx,Gy,Gz,complex.(B1),Δf,ψ,adc,t,diff(t))
    end
    return clear_rf_outside_support!(seqd,seq)
end

function simulate_case(obj, seq, sys, c, offsets, params; individual=false)
    has_dynamic = any(e->e["kind"]=="time_variation",c.effects) ||
        (any(e->e["kind"]=="motion",c.effects) && any(e->e["kind"]=="b0",c.effects))
    case_field_update!(obj,c,0.0,offsets)
    # Time-dependent fields/equilibrium are updated at each integration midpoint.
    # State persists throughout; every step delegates rotation/relaxation/acquisition
    # to the pinned KomaMRI kernels, including their native motion coordinates.
    method = individual ? CoreMRI.BlochDict() : params["sim_method"]
    seqd = discretize_case(seq,params,obj.motion,has_dynamic,c.effects)
    spoil_times = get(params,"spoil_times_s",nothing)
    block_length = has_dynamic || spoil_times !== nothing ? 1 : Inf
    ranges, rf_on = CoreMRI.get_sim_ranges(seqd;max_block_length=block_length,max_rf_block_length=block_length)
    state, obj = CoreMRI.initialize_spins_state(obj,method)
    state.z = copy(state.z) # Keep evolving Mz separate from equilibrium rho.
    backend = CoreMRI.KA.CPU()
    nthreads = individual ? 1 : min(Threads.nthreads(),length(obj))
    groupsize = CoreMRI.DEFAULT_PRECESSION_GROUPSIZE
    prealloc = CoreMRI.prealloc(method,backend,obj,state,2,groupsize)
    signal = individual ? zeros(ComplexF64,sum(seq.ADC.N),length(obj),1,1) : zeros(ComplexF64,sum(seq.ADC.N),1,nthreads)
    sample = 1
    phase_windows = get(params,"phase_windows",nothing)
    active_phase_window = 0
    next_spoil = 1
    for (range,rf) in zip(ranges,rf_on)
        step = @view seqd[range]
        mid = (first(step.t)+last(step.t))/2
        count = sum(step.ADC[2:end])
        if spoil_times !== nothing && rf
            # Destroy old transverse coherence before each excitation while
            # retaining longitudinal saturation and recovery. Composite RF
            # subpulses share one imported RF event and one reset.
            while next_spoil <= length(spoil_times) && spoil_times[next_spoil] < last(step.t)
                fill!(state.xy,0)
                next_spoil += 1
            end
        end
        if phase_windows !== nothing
            # RF-only oracle: each selected excitation starts from equilibrium,
            # independently of earlier RF. Absolute timestamps and declared
            # fields/motion remain intact. Skip idle gaps erased by that reset.
            window = findfirst(w->w[1] < last(step.t) && first(step.t) < w[2],phase_windows)
            if window === nothing
                count == 0 || error("Phase ADC outside selected echo windows")
                continue
            end
            has_dynamic && case_field_update!(obj,c,mid,offsets)
            if window != active_phase_window
                fill!(state.xy,0)
                state.z .= obj.ρ
                active_phase_window = window
            end
        else
            has_dynamic && case_field_update!(obj,c,mid,offsets)
        end
        individual || (prealloc.ΔBz .= obj.Δw ./ (2π * γ))
        count = sum(step.ADC[2:end])
        slots = sample:(sample+count-1)
        if rf
            target = individual ? @view(signal[slots,:,:,:]) : @view(signal[slots,:,:])
            CoreMRI.run_spin_excitation_parallel!(obj,step,target,state,method,groupsize,backend,prealloc;Nthreads=nthreads)
        else
            target = individual ? @view(signal[slots,:,:,:]) : @view(signal[slots,:,:])
            CoreMRI.run_spin_precession_parallel!(obj,step,target,state,method,groupsize,backend,prealloc;Nthreads=nthreads)
        end
        sample += count
    end
    return individual ? dropdims(signal;dims=4) : sum(signal;dims=3)
end

# Receive encoding accumulates each short ADC block from native per-spin Bloch
# responses. State is advanced once, independent of receive sensitivity. This
# is exactly the same density-linear operation as separate coil-weighted Bloch
# replays, with bounded memory instead of a full ADC-by-spins dictionary.
function simulate_receive_case(obj,seq,c,offsets,params,receive)
    has_dynamic = any(e->e["kind"]=="time_variation",c.effects) ||
        (any(e->e["kind"]=="motion",c.effects) && any(e->e["kind"]=="b0",c.effects))
    case_field_update!(obj,c,0.0,offsets)
    method = CoreMRI.BlochDict()
    seqd = discretize_case(seq,params,obj.motion,has_dynamic,c.effects)
    block_length = has_dynamic ? 1 : Inf
    ranges, rf_on = CoreMRI.get_sim_ranges(seqd;max_block_length=block_length,max_rf_block_length=block_length)
    state, obj = CoreMRI.initialize_spins_state(obj,method)
    state.z = copy(state.z)
    backend = CoreMRI.KA.CPU()
    groupsize = CoreMRI.DEFAULT_PRECESSION_GROUPSIZE
    prealloc = CoreMRI.prealloc(method,backend,obj,state,2,groupsize)
    size(receive,1)==length(obj) || error("Receive/spin dimension mismatch")
    signal = zeros(ComplexF64,sum(seq.ADC.N),size(receive,2))
    sample = 1
    for (range,rf) in zip(ranges,rf_on)
        step = @view seqd[range]
        has_dynamic && case_field_update!(obj,c,(first(step.t)+last(step.t))/2,offsets)
        count = sum(step.ADC[2:end])
        target = zeros(ComplexF64,count,length(obj),1,1)
        if rf
            CoreMRI.run_spin_excitation_parallel!(obj,step,target,state,method,groupsize,backend,prealloc;Nthreads=1)
        else
            CoreMRI.run_spin_precession_parallel!(obj,step,target,state,method,groupsize,backend,prealloc;Nthreads=1)
        end
        if count > 0
            @views mul!(signal[sample:sample+count-1,:],reshape(target,count,length(obj)),receive)
        end
        sample += count
    end
    return signal
end

function simulate_request(req)
    seq = deepcopy(get!(SEQUENCES,req["sequence_sha256"]) do
        read_seq(req["sequence"])
    end)
    # Receiver demodulation uses the exact original Pulseq phases in the judge;
    # prevent duplicate demodulation in KomaMRI, including diagnostic acquisitions.
    for adc in seq.ADC
        adc.ϕ = 0.0
        adc.Δf = 0.0
    end
    active_rf = [rf for rf in seq.RF if !iszero(sum(abs.(rf.A)))]
    length(active_rf) == length(req["rf_frequency_hz"]) || error("KomaMRI/Pulseq RF count mismatch")
    for (i,rf) in enumerate(active_rf)
        rf.Δf = req["rf_frequency_hz"][i]
        rf.ϕ += req["rf_phase_correction_rad"][i]
    end
    input = h5open(req["input"],"r") do f
        Dict(k=>read(f[k]) for k in keys(f))
    end
    times = Float64.(get_adc_sampling_times(seq))
    length(times)==length(input["adc_times"]) || error("KomaMRI/Pulseq ADC count mismatch")
    maximum(abs.(times-input["adc_times"])) <= 1e-9 || error("KomaMRI/Pulseq ADC time mismatch: $(maximum(abs.(times-input["adc_times"]))) s")
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
        "gpu"=>false,"precision"=>"f64","return_type"=>"mat",
        "sim_method"=>(integrator=="magnus4" ? CoreMRI.BlochMagnus4() : CoreMRI.Bloch()),
        "Δt"=>req["step_s"],"Δt_rf"=>req["rf_step_s"],
        "max_block_length"=>Inf,"max_rf_block_length"=>Inf,
        "Nthreads"=>min(Threads.nthreads(),length(obj))))
    individual = get(req,"individual",false)
    params["phase_windows"] = get(req,"phase_windows",nothing)
    params["spoil_times_s"] = get(req,"spoil_times_s",nothing)
    if individual
        params["sim_method"] = CoreMRI.BlochDict()
        params["Nthreads"] = 1
    end
    if req["demodulate_encoding"]
        isempty(req["effects"]) || error("Encoding-demodulated probes require ideal physics")
        params["sim_method"] = CoreMRI.BlochDict()
        obj.Δw .= 2π .* input["offset_hz"]
        individual = simulate_case(obj,seq,Scanner(B0=req["field_strength_T"]),c,input["offset_hz"],params;individual=true)
        signal = zeros(ComplexF64,length(times))
        for i in eachindex(times), s in eachindex(obj.x)
            phase = 2π * (input["k0"][i]*obj.x[s] + input["k1"][i]*obj.y[s] + input["k2"][i]*obj.z[s])
            signal[i] += individual[i,s,1] * cis(phase)
        end
    elseif haskey(input,"receive_weights")
        signal = simulate_receive_case(obj,seq,c,input["offset_hz"],params,input["receive_weights"])
    else
        output = simulate_case(obj,seq,Scanner(B0=req["field_strength_T"]),c,input["offset_hz"],params;individual)
        signal = individual ? output[:,:,1] : vec(output)
    end
    h5open(req["output"],"w") do f
        f["signal_real"] = real(signal)
        f["signal_imag"] = imag(signal)
    end
end

for line in eachline(stdin)
    response = try
        req = JSON.parse(line)
        if req["operation"] == "versions"
            Dict("ok"=>true,"versions"=>Dict("KomaMRICore"=>string(pkgversion(KomaMRICore)),
                "KomaMRIBase"=>string(pkgversion(KomaMRIBase)),"KomaMRIFiles"=>string(pkgversion(KomaMRIFiles)),
                "Julia"=>string(VERSION)))
        else
            redirect_stdout(stderr) do
                with_logger(NullLogger()) do
                    req["operation"]=="simulate" ? simulate_request(req) : error("Unsupported KomaMRI worker operation; reconstruction requires BART")
                end
            end
            Dict("ok"=>true)
        end
    catch e
        Dict("ok"=>false,"error"=>sprint(showerror,e,catch_backtrace()))
    end
    println(JSON.json(response)); flush(stdout)
end
