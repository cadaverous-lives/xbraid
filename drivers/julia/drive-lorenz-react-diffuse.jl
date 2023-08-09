include("../../braid/braid.jl/XBraid.jl")

using LinearAlgebra, IterativeSolvers, LinearMaps, PaddedViews
using Plots, BenchmarkTools

using .XBraid

# Matrix free Laplacian operator (0. boundary conditions)
function laplacian!(out::AbstractArray, X::AbstractArray, Δx::Real, Δy::Real; bc=0.)
    @assert size(out) == size(X)
    nx, ny = size(X)
    
    δₓ = CartesianIndex(1, 0)
    δᵧ = CartesianIndex(0, 1)
    Xp = PaddedView(bc, X, (0:nx+1, 0:ny+1))

    @views for I ∈ CartesianIndices(out)
        out[I] = (Xp[I+δₓ] - 2Xp[I] + Xp[I-δₓ])/Δx^2 + (Xp[I+δᵧ] - 2Xp[I] + Xp[I-δᵧ])/Δy^2
    end
end

struct SimApp
    # Discretization parameters
    nx::Int
    ny::Int
    lx::Float64
    ly::Float64
    Δx::Float64
    Δy::Float64

    νx::Float64  # Diffusion coefficient of X
    νy::Float64  # Diffusion coefficient of Y
    νz::Float64  # Diffusion coefficient of Z

    # system parameters
    σ::Float64
    ρ::Float64
    β::Float64

    # temporary caches
    k::Array{Float64, 3}

    # Solution history
    solution::Vector{Array{Float64, 3}}
    times::Vector{Float64}
end


function SimApp(nx::Integer; νx=.2, νy=0.2, νz=0.02, σ=10., ρ=28., β=2/3, len=(1., 1.), ny=nx) 
    lx, ly = len
    Δx = lx/(nx-1)
    Δy = ly/(ny-1)
    SimApp(nx, ny, lx, ly, Δx, Δy, νx, νy, νz, σ, ρ, β, zeros(nx, ny, 2), [], [])
end

#=
    X′ = νΔX + σ(Y - X)
    Y′ = νΔY + X(ρ - Z) - Y
    Z′ = νΔZ + XY - βZ
=#
function rhs_f!(u′::AbstractArray, u::AbstractArray, t::Real, app::SimApp)
    @views begin 
        X = u[:, :, 1]
        Y = u[:, :, 2]
        Z = u[:, :, 3]
        X′ = u′[:, :, 1]
        Y′ = u′[:, :, 2]
        Z′ = u′[:, :, 3]
    end
    σ, ρ, β = app.σ, app.ρ, app.β

    laplacian!(X′, X, app.Δx, app.Δy; bc=0.)
    laplacian!(Y′, Y, app.Δx, app.Δy; bc=0.)
    laplacian!(Z′, Z, app.Δx, app.Δy; bc=0.)
    X′ .*= app.νx
    Y′ .*= app.νy
    Z′ .*= app.νz
    @. X′ += σ*(Y - X)
    @. Y′ += X*(ρ - Z) - Y
    @. Z′ += X*Y - β*Z
    nothing
end

function rhs_f(u, t, app)
    u′ = similar(u)
    rhs_f!(u′, u, t, app)
    return u′
end

#=
    J = [νΔ - σI  σI       0;
         ρI - Z   νΔ - I  -X
         Y        X        νΔ - βI]
        
=#
function rhs_Jvp!(Jv::AbstractArray, u::AbstractArray, t::Real, v::AbstractArray, app::SimApp)
    @views begin
        X = u[:, :, 1]
        Y = u[:, :, 2]
        Z = u[:, :, 3]
        Jv_X = Jv[:, :, 1]
        Jv_Y = Jv[:, :, 2]
        Jv_Z = Jv[:, :, 3]
        v_X = v[:, :, 1]
        v_Y = v[:, :, 2]
        v_Z = v[:, :, 3]
    end 
    
    σ, ρ, β = app.σ, app.ρ, app.β

    laplacian!(Jv_X, v_X, app.Δx, app.Δy; bc=0.)
    laplacian!(Jv_Y, v_Y, app.Δx, app.Δy; bc=0.)
    laplacian!(Jv_Z, v_Z, app.Δx, app.Δy; bc=0.)
    Jv_X .*= app.νx
    Jv_Y .*= app.νy
    Jv_Z .*= app.νz
    @. Jv_X += -σ*v_X + σ*v_Y
    @. Jv_Y += (ρ - Z)*v_X - v_Y - X*v_Z
    @. Jv_Z += Y*v_X + X*v_Y - β*v_Z
    nothing
end

function rhs_Jvp(u, t, v, app)
    Jv = similar(v)
    rhs_Jvp!(Jv, u, t, v, app)
    return Jv
end

function rhs_J(u::Array{<:Real, 3}, t::Real, app::SimApp)
    LinearMap((Jv, v) -> rhs_Jvp!(reshape(Jv, size(u)), u, t, reshape(v, size(u)), app), length(u), ismutating=true)
end

# Inexact Newton's method for solving F(u) = 0, using GMRES
@views function Newton!(guess::AbstractArray, F!::Function, J_u::Function; maxiter=20, otol=1e-6, itol=1e-6, verbose=false)
    x = vec(guess)
    δ = zeros(length(x))
    r = similar(x)
    verbose && println("Newton solve:")
    for i in 1:maxiter
        F!(reshape(r, size(guess)), guess)

        # tolerance checking
        r_norm = norm(r)
        verbose && print("$i: ||F(u)|| = $r_norm\t")
        if (r_norm < otol || (i > 1 && norm(δ) < otol))
            verbose && println("tol. reached")
            return guess
        end

        J = J_u(guess)
        if verbose
            δ, hist = IterativeSolvers.bicgstabl!(δ, J, r; log=true, reltol=1e-6)
        else
            δ = IterativeSolvers.bicgstabl!(δ, J, r; log=false, reltol=1e-6)
        end
        verbose && println("GMRES history: $hist")

        x .-= δ
    end
    return guess
end

function my_init(app::SimApp, t::Real)
    u = zeros(app.nx, app.nx, 3)
    # perturbation
    u .+= 0.1randn(size(u))
    return u
end

function euler(app::SimApp, u::Array{<:Real, 3}, tstart::Real, tstop::Real; verbose=false)
    Δt = tstop - tstart

    function res!(r_v, k)
        r = reshape(r_v, size(k))
        rhs_f!(r, u .+ k, tstart + Δt, app)
        @. r = k - Δt * r
    end

    function res_J(k)
        I = LinearMaps.UniformScalingMap(1., length(k))
        J = rhs_J(u .+ k, tstart + Δt, app)
        return I - Δt * J
    end

    k = zeros(size(u))
    Newton!(k, res!, res_J; verbose=verbose, otol=1e-6)
    u .+= k 
end

function sdirk2(app::SimApp, u::Array{<:Real, 3}, tstart::Real, tstop::Real; verbose=false)
    Δt = tstop - tstart

    α = 1 - √2/2

    function res1!(r_v, k1)
        r = reshape(r_v, size(k1))
        rhs_f!(r, u .+ α*k1, tstart + α*Δt, app)
        @. r = k1 - Δt * r
    end

    function res1_J(k)
        I = LinearMaps.UniformScalingMap(1., length(k))
        J = rhs_J(u .+ α*k, tstart + α*Δt, app)
        return I - Δt * J
    end

    verbose && println("SDIRK2: Stage 1")
    k1 = zeros(size(u))
    Newton!(k1, res1!, res1_J; verbose=verbose, otol=1e-6)

    function res2!(r_v, u1)
        r = reshape(r_v, size(u1))
        rhs_f!(r, u1, tstart + Δt, app)
        @. r = u1 - (u + (1-α)k1 + α*Δt*r)
    end

    function res2_J(u1)
        I = LinearMaps.UniformScalingMap(1., length(u1))
        J = rhs_J(u1, tstart + Δt, app)
        return I - α*Δt*J
    end

    verbose && println("SDIRK2: Stage 2")
    u1 = deepcopy(u)
    u1 .+= k1
    Newton!(u1, res2!, res2_J; verbose=verbose, otol=1e-6)
    u .= u1
end

function test()
    app = SimApp(128; len=(16., 16.))
    u = my_init(app, 0.)
    u .+= randn(size(u))

    # Check that the Jvp is correct
    v = randn(size(u))
    Jv = rhs_Jvp(u, 0., v, app)
    Jv2 = (rhs_f(u + 1e-6v, 0., app) - rhs_f(u, 0., app)) / 1e-6
    println("Jvp error: ", norm(Jv - Jv2)/norm(Jv2))

    # Check that Newton is converging
    # euler(app, u, 0., .01; verbose=true)
    sdirk2(app, u, 0., .01; verbose=true)
    heatmap(u[:, :, 3])
end
test()

function main(;nx=64, nt=1000, tstop=10., len=16., νx=0.1, νy=0.1, νz=0.01)
    app = SimApp(nx; νx=νx, νy=νy, νz=νz, len=(len, len))
    u = my_init(app, 0.)
    t = 0.
    Δt = tstop / nt
    anim = @animate for i in 1:nt÷4
        println("t = $(t+Δt)")
        for _ in 1:4
            sdirk2(app, u, t, t + Δt)
            t += Δt
        end
        push!(app.solution, deepcopy(u))
        heatmap(u[:, :, 1])
    end

    gif(anim, "lorenz-rd.gif", fps=30)
    return app
end
# main();