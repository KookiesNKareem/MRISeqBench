vec3(v) = vcat(Float64.(v), zeros(3-length(v)))
sinusoid(e,t) = sin(2π*t/e["period_s"] + deg2rad(get(e,"phase_deg",0)))
function rotation(angles)
    a,b,c = deg2rad.(angles)
    rx = [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)]
    ry = [cos(b) 0 sin(b); 0 1 0; -sin(b) 0 cos(b)]
    rz = [cos(c) -sin(c) 0; sin(c) cos(c) 0; 0 0 1]
    return rz*ry*rx
end

function coordinates(obj,e,t)
    p = hcat(obj.x,obj.y,obj.z)
    model = e["model"]
    if model == "rigid_sinusoidal"
        s = sinusoid(e,t)
        return p * rotation([0,0,e["rotation_amplitude_deg"]*s])' .+
               (vec3(e["translation_amplitude_mm"])*s/1000)'
    elseif model == "affine_sinusoidal"
        return p .* (1 .+ vec3(e["strain_amplitude"])*sinusoid(e,t))'
    elseif model == "rigid_keyframes"
        poses = e["poses"]
        i = max(1, searchsortedlast([q["time_s"] for q in poses], t))
        q = poses[i]
        tr = vec3(q["translation_mm"]); ang = vec3(q["rotation_xyz_deg"])
        if get(e,"interpolation","previous") == "linear" && i < length(poses)
            q2 = poses[i+1]; f = clamp((t-q["time_s"])/(q2["time_s"]-q["time_s"]),0,1)
            tr = (1-f)*tr + f*vec3(q2["translation_mm"])
            ang = (1-f)*ang + f*vec3(q2["rotation_xyz_deg"])
        end
        return p*rotation(ang)' .+ (tr/1000)'
    end
    error("Unsupported motion model: $model")
end

function spatial_field(e,p)
    isb0 = e["kind"] == "b0"
    model = e["model"]
    baseline = isb0 ? get(e,"offset_hz",0) : get(e,"baseline_scale",get(e,"center_scale",get(e,"scale",1)))
    model == "constant" && return fill(Float64(baseline),size(p,1))
    if model == "linear"
        grad = vec3(e[isb0 ? "gradient_hz_per_m" : "gradient_scale_per_m"])
        return baseline .+ p*grad
    elseif model == "gaussian"
        center = vec3(e["center_mm"])/1000
        sigma = vcat(Float64.(e["sigma_mm"])/1000, ones(3-length(e["sigma_mm"])))
        exponent = sum(((p .- center') ./ sigma').^2,dims=2)
        return baseline .+ e[isb0 ? "amplitude_hz" : "amplitude_scale"] .* exp.(-vec(exponent)/2)
    end
    error("Map fields require a dedicated registered field loader")
end

"""Exact-time native phantom snapshot and per-spin transmit scale."""
function snapshot(c::PhantomCase,t::Real=0.0)
    isfinite(t) && t >= 0 || error("Time must be finite and nonnegative")
    obj = deepcopy(c.phantom); obj.motion = NoMotion()
    for e in c.effects
        e["kind"] == "motion" || continue
        p = coordinates(obj,e,t); obj.x=p[:,1]; obj.y=p[:,2]; obj.z=p[:,3]
    end
    p = hcat(obj.x,obj.y,obj.z)
    b0 = zeros(length(obj)); b1 = ones(length(obj))
    for e in c.effects
        kind = e["kind"]
        if kind == "b0"
            b0 .+= spatial_field(e,p)
        elseif kind == "b1"
            b1 .*= spatial_field(e,p)
        elseif kind == "time_variation"
            s = sinusoid(e,t); target = e["target"]
            if target == "b0_offset"
                b0 .+= e["amplitude_hz"]*s
            elseif target == "b1_scale"
                b1 .*= 1+e["amplitude_scale"]*s
            elseif target == "proton_density"
                obj.ρ[c.labels .== e["material"]] .*= 1+e["amplitude_scale"]*s
            else
                error("Unsupported time variation target: $target")
            end
        end
    end
    obj.Δw .= 2π .* b0
    all(b1 .> 0) || error("B1 scale must remain positive")
    return obj,b1
end

function native_motion!(c)
    obj = c.phantom
    for e in c.effects
        e["kind"] == "motion" || continue
        if e["model"] == "rigid_sinusoidal"
            # A periodic native time curve with 1024 intervals per cycle.
            t=collect(range(0.0,e["period_s"],length=1025))
            tc=TimeCurve(t=t,t_unit=sinusoid.(Ref(e),t),periodic=true)
            a=vec3(e["translation_amplitude_mm"])/1000
            obj.motion=MotionList(rotate(0.0,0.0,Float64(e["rotation_amplitude_deg"]),tc;center=(0.0,0.0,0.0)),
                                  translate(a...,tc))
        else
            if e["model"] == "rigid_keyframes"
                poses=e["poses"]; t=Float64[q["time_s"] for q in poses]
                if get(e,"interpolation","previous") == "previous"
                    t=sort(unique(vcat(t,[prevfloat(v) for v in t[2:end]])))
                elseif length(t)>1
                    t=sort(unique(vcat([collect(range(a,b,length=1025)) for (a,b) in zip(t[1:end-1],t[2:end])]...)))
                end
                length(t)==1 && push!(t,nextfloat(t[1]))
                periodic=false
            else
                t=collect(range(0.0,e["period_s"],length=1025)); periodic=true
            end
            ps=[coordinates(obj,e,v) for v in t]
            dx=hcat([q[:,1]-obj.x for q in ps]...)
            dy=hcat([q[:,2]-obj.y for q in ps]...)
            dz=hcat([q[:,3]-obj.z for q in ps]...)
            tc=TimeCurve(t=t,t_unit=collect(range(0.0,1.0,length=length(t))),periodic=periodic)
            obj.motion=path(dx,dy,dz,tc)
        end
    end
    return c
end
