import KomaMRI
using JSON, HDF5
import KomaMRIBase
using KomaMRIBase: CardiacSignal, resolve_triggers

function resolve_simulation_physio(seq, rr_interval_s=1.0)
    heart_rate_hz = inv(Float64(rr_interval_s))
    resolved = resolve_triggers(seq, CardiacSignal(; heart_rate=heart_rate_hz, first_peak=0.0))
    original_duration_s = sum(seq.DUR)
    resolved_duration_s = sum(resolved.DUR)
    metadata = Dict(
        "resolver" => "KomaMRIBase.resolve_triggers/CardiacSignal",
        "heart_rate_hz" => heart_rate_hz,
        "first_peak_s" => 0.0,
        "original_duration_s" => original_duration_s,
        "resolved_duration_s" => resolved_duration_s,
        "inserted_wait_s" => resolved_duration_s - original_duration_s,
    )
    return resolved, metadata
end

function reconstruct_files(signal, seq, sys, output; reconstruct=KomaMRI.reconstruct_with_labels)
    metadata = Dict{String,Any}(
        "api" => "KomaMRI.reconstruct_with_labels",
        "method" => "direct",
        "diagnostic" => true,
        "sequence_definitions" => seq.DEF,
    )
    undefined_blocks = [block for block in axes(seq.RF, 2)
                        if any(rf -> KomaMRIBase.is_on(rf) &&
                               rf.use isa KomaMRIBase.Undefined, seq.RF[:, block])]
    roles_ok = isempty(undefined_blocks)
    role_detail = roles_ok ? "All active RF events declare their use." :
        "Undefined RF use in blocks $(join(undefined_blocks, ", ")). Set use=Excitation(), Refocusing(), Inversion(), Saturation(), Preparation(), or Other() as appropriate. RF roles and centers determine the reconstruction trajectory."
    metadata["checks"] = [Dict("name" => "rf_roles", "passed" => roles_ok,
                              "detail" => role_detail)]
    try
        roles_ok || error(role_detail)
        raw = KomaMRI.KomaMRICore.signal_to_raw_data(signal, seq; sys)
        result = reconstruct(raw; rec_params=Dict(:reco => "direct"))
        images = []
        h5open(joinpath(output, "reconstruction.h5"), "w") do file
            for (index, entry) in enumerate(result.images)
                name = "image_$(lpad(index, 3, '0'))"
                image = Array(entry.image)
                group = create_group(file, name)
                # HDF5.jl reverses dimensions for C readers; expose the documented
                # image axes in the same order to Python/h5py and Julia clients.
                order = ndims(image):-1:1
                group["image_real", compress=3] = permutedims(real.(image), order)
                group["image_imag", compress=3] = permutedims(imag.(image), order)
                axes = String.(KomaMRI.AxisArrays.axisnames(entry.image))
                push!(images, Dict(
                    "group" => name, "shape" => collect(size(image)), "axes" => collect(axes),
                    "axis_values" => [collect(values) for values in KomaMRI.AxisArrays.axisvalues(entry.image)],
                    "labels" => Dict(pairs(entry.labels)), "source" => Dict(pairs(entry.source)),
                    "trajectory_scale" => entry.trajectory_scale,
                ))
            end
        end
        metadata["status"] = "completed"
        metadata["policy"] = Dict(pairs(result.policy))
        metadata["images"] = images
    catch exception
        # A malformed or unencoded acquisition can still yield useful raw signal.
        metadata["status"] = "failed"
        metadata["error"] = sprint(showerror, exception)
    end
    open(joinpath(output, "reconstruction.json"), "w") do io
        JSON.print(io, metadata)
    end
    return metadata
end
