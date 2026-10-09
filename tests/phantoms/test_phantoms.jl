using Test, KomaMRIBase, KomaMRIFiles, JSON, HDF5
include(joinpath(@__DIR__, "..", "..", "src", "mriseqbench", "phantoms", "loader.jl"))
using .MRISeqBenchPhantoms
case(ref,effects=Any[]) = load_case(Dict("object"=>Dict("phantom"=>ref),"physics"=>Dict("effects"=>effects),"seed"=>42))
@testset "Registered native phantoms" begin
    for ref in keys(MRISeqBenchPhantoms.REGISTRY)
        obj,labels=build_phantom(ref)
        @test obj isa Phantom
        @test !isempty(obj.x)
        @test all(obj.ρ .> 0)
        @test all(obj.T1 .>= obj.T2)
        @test length(labels)==length(obj)
        @test maximum(abs.(obj.x))<0.1
        obj2,labels2=build_phantom(ref)
        @test obj.x==obj2.x && obj.ρ==obj2.ρ && labels==labels2
    end
    obj,labels=build_phantom("tissue_discs@1")
    @test Set(labels)==Set(["wm","gm","csf","fat"])
    @test all(obj.T1[labels.=="wm"].==0.8)
    @test all(obj.T2[labels.=="csf"].==2.0)
    @test_throws ErrorException build_phantom("missing@1")
end
@testset "Fields and exact snapshots" begin
    b0=Dict("kind"=>"b0","model"=>"linear","offset_hz"=>10,"gradient_hz_per_m"=>[1000,0])
    b1=Dict("kind"=>"b1","model"=>"constant","scale"=>0.8)
    c=case("uniform_disc@1",[b0,b1]); obj,scale=snapshot(c)
    @test obj.Δw ≈ 2π.*(10 .+ 1000 .*obj.x)
    @test all(scale.==0.8)
    move=Dict("kind"=>"motion","model"=>"rigid_sinusoidal","translation_amplitude_mm"=>[3,0],"rotation_amplitude_deg"=>0,"period_s"=>1)
    c=case("uniform_disc@1",[b0,b1,move]); obj,scale=snapshot(c,0.25)
    @test obj.x ≈ c.phantom.x .+ 0.003
    @test obj.Δw ≈ 2π.*(10 .+ 1000 .*obj.x)
    c=case("tissue_discs@1",[Dict("kind"=>"time_variation","model"=>"sinusoidal","target"=>"proton_density","material"=>"wm","amplitude_scale"=>0.1,"period_s"=>2)])
    obj,_=snapshot(c,0.5)
    @test all(obj.ρ[c.labels.=="wm"].≈0.77)
    @test all(obj.ρ[c.labels.=="gm"].==0.8)
    for (target,key,amplitude) in (("b0_offset","amplitude_hz",40),("b1_scale","amplitude_scale",0.1))
        c=case("uniform_disc@1",[Dict("kind"=>"time_variation","model"=>"sinusoidal","target"=>target,key=>amplitude,"period_s"=>2)])
        obj,b=snapshot(c,0.5)
        @test target=="b0_offset" ? all(obj.Δw.≈2π*40) : all(b.≈1.1)
    end
end
@testset "Native motion and saved artifacts" begin
    effects=[Dict("kind"=>"motion","model"=>"rigid_sinusoidal","translation_amplitude_mm"=>[3,0],"rotation_amplitude_deg"=>2,"period_s"=>1,"phase_deg"=>0)]
    c=case("uniform_disc@1",effects)
    mktempdir() do out
        m=materialize(c,out)
        obj=read_phantom(joinpath(out,"phantom.phantom"))
        @test obj isa Phantom
        @test obj.ρ==c.phantom.ρ
        for t in (0.0,0.25,0.5,0.75,1.25)
            actual=get_spin_coords(obj.motion,obj.x,obj.y,obj.z,reshape([t],1,:))
            expected,_=snapshot(c,t)
            @test vec(actual[1]) ≈ expected.x atol=1e-10
            @test vec(actual[2]) ≈ expected.y atol=1e-10
        end
        h5open(joinpath(out,"fields.h5"),"r") do f
            @test read(f["b1_scale_t0"])==ones(length(obj))
            @test read(f["material_labels"])==c.labels
        end
        @test m["spins"]==length(obj)
        @test_throws ErrorException materialize(c,out)
    end
    poses=[Dict("time_s"=>0,"translation_mm"=>[0,0,0],"rotation_xyz_deg"=>[0,0,0]),Dict("time_s"=>1,"translation_mm"=>[3,0,0],"rotation_xyz_deg"=>[0,0,2])]
    c=case("uniform_disc@1",[Dict("kind"=>"motion","model"=>"rigid_keyframes","interpolation"=>"previous","poses"=>poses)])
    MRISeqBenchPhantoms.native_motion!(c)
    for t in (0.0,0.5,prevfloat(1.0),1.0,2.0)
        actual=get_spin_coords(c.phantom.motion,c.phantom.x,c.phantom.y,c.phantom.z,reshape([t],1,:))
        expected,_=snapshot(c,t)
        @test vec(actual[1]) ≈ expected.x atol=1e-10
    end
    c=case("uniform_disc@1",[Dict("kind"=>"motion","model"=>"affine_sinusoidal","strain_amplitude"=>[0.05,-0.03],"period_s"=>1)])
    MRISeqBenchPhantoms.native_motion!(c)
    actual=get_spin_coords(c.phantom.motion,c.phantom.x,c.phantom.y,c.phantom.z,reshape([0.25],1,:))
    @test vec(actual[1]) ≈ 1.05 .*c.phantom.x
    @test vec(actual[2]) ≈ 0.97 .*c.phantom.y
end
@testset "Static fields survive native file serialization" begin
    effects=[Dict("kind"=>"b0","model"=>"constant","offset_hz"=>100),Dict("kind"=>"b1","model"=>"constant","scale"=>0.8)]
    c=case("uniform_disc@1",effects)
    mktempdir() do out
        m=materialize(c,out)
        p=read_phantom(joinpath(out,"phantom.phantom"))
        @test all(p.Δw.≈2π*100)
        @test all(p.T2s.==1_000_000)
        @test !m["requires_field_updates"]
        h5open(joinpath(out,"fields.h5"),"r") do f
            @test all(read(f["b1_scale_t0"]).==0.8)
        end
    end
    @test_throws ErrorException case("uniform_disc@1",[Dict("kind"=>"other","model"=>"constant")])
    @test_throws ErrorException snapshot(c,NaN)
    @test_throws ErrorException snapshot(c,-1)
end
@testset "Registry bounds and smooth fields" begin
    for (ref,meta) in MRISeqBenchPhantoms.REGISTRY
        p,labels=build_phantom(ref)
        @test Set(labels)==Set(meta["materials"])
        coords=(p.x,p.y,p.z)
        for axis in 1:meta["dimensions"]
            lo,hi=meta["bounds_mm"][axis]./1000
            @test minimum(coords[axis])>=lo && maximum(coords[axis])<=hi
        end
    end
    c=case("uniform_disc@1",[Dict("kind"=>"b1","model"=>"linear","center_scale"=>1,"gradient_scale_per_m"=>[0,2])])
    p,b=snapshot(c)
    @test b ≈ 1 .+ 2 .*p.y
    c=case("uniform_disc@1",[Dict("kind"=>"b1","model"=>"gaussian","baseline_scale"=>1,"amplitude_scale"=>-0.35,"center_mm"=>[0,20],"sigma_mm"=>[40,25])])
    p,b=snapshot(c)
    @test b ≈ 1 .- 0.35 .*exp.(-0.5 .* ((p.x./0.04).^2 .+ ((p.y.-0.02)./0.025).^2))
    c=case("uniform_disc@1",[Dict("kind"=>"b0","model"=>"gaussian","offset_hz"=>0,"amplitude_hz"=>150,"center_mm"=>[30,0],"sigma_mm"=>[20,35])])
    p,_=snapshot(c)
    @test p.Δw ≈ 2π*150 .*exp.(-0.5 .* (((p.x.-0.03)./0.02).^2 .+ (p.y./0.035).^2))
end

@testset "3D field and motion controls" begin
    b0=Dict("kind"=>"b0","model"=>"linear","offset_hz"=>0,"gradient_hz_per_m"=>[1000,0,300])
    b1=Dict("kind"=>"b1","model"=>"linear","center_scale"=>1,"gradient_scale_per_m"=>[0,2,0.5])
    motion=Dict("kind"=>"motion","model"=>"rigid_sinusoidal","translation_amplitude_mm"=>[3,0,2],"rotation_amplitude_deg"=>0,"period_s"=>1)
    c=case("uniform_sphere@1",[b0,b1,motion])
    obj,scale=snapshot(c,0.25)
    @test obj.z ≈ c.phantom.z .+ 0.002
    @test obj.Δw ≈ 2π.*(1000 .* obj.x .+ 300 .* obj.z)
    @test scale ≈ 1 .+ 2 .*obj.y .+ 0.5 .*obj.z
    mktempdir() do out
        materialize(c,out)
        saved=read_phantom(joinpath(out,"phantom.phantom"))
        coords=get_spin_coords(saved.motion,saved.x,saved.y,saved.z,reshape([0.25],1,:))
        @test vec(coords[3]) ≈ obj.z
    end
end
