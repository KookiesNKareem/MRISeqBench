# One process per trial. Only authoring libraries are loaded; no task inputs or
# judge state are used by the fixed warmup. The protocol stays inside the author
# container, separate from the trusted host transport.
module MRISeqBenchJuliaSession

using TOML
using Revise
using KomaMRIBase, KomaMRIFiles

const directory = "/tmp/mriseqbench-julia"
const collect_requests = Base.JLOptions().heap_size_hint > 0

Base.@noinline function warmup(pulse_designer, unitful)
    @assert isdefined(KomaMRIBase, :PulseDesigner) == pulse_designer
    @assert (Base.find_package("Unitful") !== nothing) == unitful
    sys = Scanner()
    rf = RF((π / 6) / (2π * γ * 1e-3), 1e-3, 0.0, sys.limits.RF_dead_time;
            use=Excitation())
    adc = ADC(32, 31e-4, sys.limits.ADC_dead_time + 0.5e-4)
    if pulse_designer
        rf = PulseDesigner.make_block_pulse(π / 6; duration=1e-3, sys, use=Excitation())
        adc = PulseDesigner.make_adc(32; duration=3.2e-3, sys)
    end
    if unitful
        @eval using Unitful
        rf, adc = Core.eval(@__MODULE__, quote
            let sys = $sys
                (PulseDesigner.make_block_pulse(π / 6; duration=1u"ms", sys, use=Excitation()),
                 PulseDesigner.make_adc(32; duration=3.2u"ms", sys))
            end
        end)
    end
    seq = Sequence(sys)
    addblock!(seq, rf, Duration(dur(rf, sys)))
    addblock!(seq, adc, Duration(dur(adc, sys)))
    # Also compile a generic low-level gradient constructor in every arm.
    addblock!(seq, Duration(1.2e-3); x=Grad(1e-3, 1e-3, 1e-4, 1e-4, 0.0))
    mktempdir() do folder
        path = joinpath(folder, "warmup.seq")
        write_seq(seq, path; sys, verbose=false)
        read_seq(path)
    end
    # Exercise Revise and the execution path before readiness, not on the first
    # agent call. Warmup definitions live outside the agent's Main namespace.
    path = joinpath(directory, "warmup.jl")
    write(path, "module MRISeqBenchReviseWarmup; value() = 1; end\n")
    Base.invokelatest(Revise.includet, @__MODULE__, path)
    Revise.revise(; throw=true)
    write(path, "module MRISeqBenchReviseWarmup; value() = 2; end\n")
    revised = false
    for _ in 1:40
        sleep(0.05)
        Revise.revise(; throw=true)
        value = getfield(getfield(@__MODULE__, :MRISeqBenchReviseWarmup), :value)
        if Base.invokelatest(value) == 2
            revised = true
            break
        end
    end
    @assert revised "Revise warmup did not observe a method edit"
end

Base.@noinline function release_memory()
    GC.gc(true)
    if collect_requests
        # A second collection releases finalized native compiler allocations.
        # glibc can otherwise retain those free pages in a long-lived REPL.
        GC.gc(true)
        if Sys.islinux()
            try
                ccall(:malloc_trim, Cint, (Csize_t,), 0)
            catch
                # Non-glibc Linux runtimes need not expose malloc_trim.
            end
        end
    end
end

Base.@noinline function execute_request(folder)
    request = TOML.parsefile(joinpath(folder, "request.toml"))
    code = 0
    open(joinpath(folder, "stdout"), "w") do out
        open(joinpath(folder, "stderr"), "w") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    try
                        empty!(ARGS)
                        append!(ARGS, request["args"])
                        cd(request["cwd"]) do
                            Revise.revise(; throw=true)
                            if request["mode"] == "eval"
                                # Match a REPL working directory rather than inheriting
                                # this server script's SOURCE_PATH through serve().
                                source = joinpath(pwd(), "agent-eval")
                                task_local_storage(:SOURCE_PATH, source) do
                                    Base.invokelatest(Base.include_string, Main, request["value"], source)
                                end
                            elseif request["mode"] == "file"
                                Base.invokelatest(Base.include, Main, abspath(request["value"]))
                            else
                                error("unknown Julia session request mode")
                            end
                        end
                    catch exception
                        code = 1
                        showerror(stderr, exception, catch_backtrace())
                        println(stderr)
                    end
                end
            end
        end
    end
    # With an explicit low-memory GC target, release transient allocations
    # before another request or a separate simulator needs the RAM.
    collect_requests && release_memory()
    write(joinpath(folder, "done.tmp"), string(code))
    mv(joinpath(folder, "done.tmp"), joinpath(folder, "done"))
end

Base.@noinline function warm_main()
    # Exercise the default Main includet path too, while keeping definitions
    # inside the private warmup module. Its compilation must precede the timer.
    main_path = joinpath(directory, "warmup-main.jl")
    write(main_path, "MRISeqBenchJuliaSession.MRISeqBenchReviseWarmup.value() = 3\n")
    mktempdir(directory) do folder
        request = Dict("mode"=>"eval", "value"=>"includet($(repr(main_path)))", "cwd"=>pwd(), "args"=>String[])
        open(joinpath(folder, "request.toml"), "w") do io
            TOML.print(io, request)
        end
        execute_request(folder)
        @assert read(joinpath(folder, "done"), String) == "0" read(joinpath(folder, "stderr"), String)
    end
    write(main_path, "MRISeqBenchJuliaSession.MRISeqBenchReviseWarmup.value() = 4\n")
    revised = false
    for _ in 1:40
        sleep(0.05)
        Revise.revise(; throw=true)
        value = getfield(getfield(@__MODULE__, :MRISeqBenchReviseWarmup), :value)
        if Base.invokelatest(value) == 4
            revised = true
            break
        end
    end
    @assert revised "Main Revise warmup did not observe a method edit"
end

function serve()
    pulse_designer, unitful = ARGS .== "true"
    mkpath(directory)
    Core.eval(Main, :(using Revise, KomaMRIBase, KomaMRIFiles))
    unitful && Core.eval(Main, :(using Unitful))
    warmup(pulse_designer, unitful)
    # Compile the same request handler used for agent work, then remove its
    # scratch artifacts. This defines no sequence-design solution in Main.
    mktempdir(directory) do folder
        open(joinpath(folder, "request.toml"), "w") do io
            TOML.print(io, Dict("mode"=>"eval", "value"=>"nothing", "cwd"=>pwd(), "args"=>String[]))
        end
        execute_request(folder)
        @assert read(joinpath(folder, "done"), String) == "0"
    end
    warm_main()
    release_memory()
    ready = "{\"mode\":\"persistent\",\"pid\":$(getpid()),\"julia_version\":\"$VERSION\",\"revise_version\":\"$(pkgversion(Revise))\",\"warmup\":[\"authoring\",\"pulseq_write_read\",\"revise\",\"revise_edit\"]}\n"
    write(joinpath(directory, "ready.tmp"), ready)
    mv(joinpath(directory, "ready.tmp"), joinpath(directory, "ready.json"))
    while true
        for folder in readdir(directory; join=true)
            if isfile(joinpath(folder, "request.toml")) && !isfile(joinpath(folder, "done"))
                execute_request(folder)
            end
        end
        sleep(0.02)
    end
end

end

MRISeqBenchJuliaSession.serve()
