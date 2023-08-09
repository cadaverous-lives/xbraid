include("../../braid/braid.jl/XBraid.jl")

using LinearAlgebra, IterativeSolvers, LinearMaps, PaddedViews
using Plots, BenchmarkTools

using .XBraid

# Matrix free Laplacian operator (0. boundary conditions)
function laplacian!(out::AbstractMatrix, X::AbstractMatrix, Δx::Real, Δy::Real; bc=0.)
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
    A::Float64  # Concentration of A
    B::Float64  # Concentration of B

    # temporary caches
    k::Array{Float64, 3}

    # Solution history
    solution::Vector{Array{Float64, 3}}
    times::Vector{Float64}
end


function SimApp(nx::Integer; νx=.2, νy=0.02, A=1., B=3., len=(1., 1.), ny=nx) 
    lx, ly = len
    Δx = lx/(nx-1)
    Δy = ly/(ny-1)
    SimApp(nx, ny, lx, ly, Δx, Δy, νx, νy, A, B, zeros(nx, ny, 2), [], [])
end

# function fft_laplacian!(out::AbstractArray, u::AbstractArray, app::SimApp)
#     @views X = u[:, :, 1]
#     @views Y = u[:, :, 1]
#     app.X̂ .= app.F̂ * X
#     app.Ŷ .= app.F̂ * Y

#     app.X̂ .*= -1app.κ.^2
#     app.Ŷ .*= -1app.κ.^2

#     @views out[:, :, 1] .= app.F̂ \ app.X̂
#     @views out[:, :, 2] .= app.F̂ \ app.Ŷ
#     nothing
# end


#=
    X′ = νΔX + A + X^2 Y - (B + 1)X
    Y′ = νΔY + BX - X^2 Y
=#
function rhs_f!(u′::AbstractArray, u::AbstractArray, t::Real, app::SimApp)
    @views begin 
        X = u[:, :, 1]
        Y = u[:, :, 2]
        X′ = u′[:, :, 1]
        Y′ = u′[:, :, 2]
    end
    A, B = app.A, app.B

    laplacian!(X′, X, app.Δx, app.Δy; bc=A)
    laplacian!(Y′, Y, app.Δx, app.Δy; bc=B/A)
    # laplacian!(X′, X, app.Δx, app.Δy; bc=0.)
    # laplacian!(Y′, Y, app.Δx, app.Δy; bc=0.)
    X′ .*= app.νx
    Y′ .*= app.νy
    @. X′ += A + X^2 * Y - (B + 1)X
    @. Y′ += B * X - X^2 * Y
    nothing
end

function rhs_f(u, t, app)
    u′ = similar(u)
    rhs_f!(u′, u, t, app)
    return u′
end

#=
    J = [νΔ + 2X * Y - (B + 1)  X^2    ;
         B - 2X * Y             νΔ - X^2]
=#
function rhs_Jvp!(Jv::AbstractArray, u::AbstractArray, t::Real, v::AbstractArray, app::SimApp)
    @views begin
        X = u[:, :, 1]
        Y = u[:, :, 2]
        Jv_X = Jv[:, :, 1]
        Jv_Y = Jv[:, :, 2]
        v_X = v[:, :, 1]
        v_Y = v[:, :, 2]
    end 
    A, B = app.A, app.B

    laplacian!(Jv_X, v_X, app.Δx, app.Δy; bc=0.)
    laplacian!(Jv_Y, v_Y, app.Δx, app.Δy; bc=0.)
    Jv_X .*= app.νx
    Jv_Y .*= app.νy
    @. Jv_X += (2X*Y - (B + 1)) * v_X + X^2 * v_Y
    @. Jv_Y +=       (B - 2X*Y) * v_X - X^2 * v_Y
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
    for i in 1:maxiter
        verbose && println("Newton iteration $i")
        F!(reshape(r, size(guess)), guess)

        # tolerance checking
        r_norm = norm(r)
        verbose && println("||F(u)|| = $r_norm")
        r_norm           < otol && return guess
        i > 1 && norm(δ) < otol && return guess

        J = J_u(guess)
        if verbose
            δ, hist = IterativeSolvers.bicgstabl!(δ, J, r; log=true, reltol=1e-3)
        else
            δ = IterativeSolvers.bicgstabl!(δ, J, r; log=false, reltol=1e-3)
        end
        verbose && println("GMRES history: $hist")

        x .-= δ
    end
    return guess
end

function my_init(app::SimApp, t::Real)
    u = zeros(app.nx, app.nx, 2)
    # fixed point
    @views begin
        @. u[:, :, 1] = app.A
        @. u[:, :, 2] = app.B/app.A
    end
    # perturbation
    u .+= 0.1randn(size(u))
    return u
end

function my_step(app::SimApp, u::Array{<:Real, 3}, tstart::Real, tstop::Real; verbose=false)
    Δt = tstop - tstart

    function res!(r_v, k)
        r = reshape(r_v, size(k))
        rhs_f!(r, u .+ k, tstart + Δt, app)
        @. r = k - Δt * r
    end

    function res_J(k)
        I = LinearMaps.UniformScalingMap(1., length(k))
        J = rhs_J(u + k, tstart + Δt, app)
        return I - Δt * J
    end

    k = zeros(size(u))
    Newton!(k, res!, res_J; verbose=verbose)
    u .+= k
end

function test()
    app = SimApp(64)
    u = my_init(app, 0.)
    u .+= randn(size(u))

    # Check that X = A, Y = B/A is a fixed point
    println("Fixed point error: ", norm(rhs_f(u, 0., app)))

    # Check that the Jvp is correct
    v = randn(size(u))
    Jv = rhs_Jvp(u, 0., v, app)
    Jv2 = (rhs_f(u + 1e-6v, 0., app) - rhs_f(u, 0., app)) / 1e-6
    println("Jvp error: ", norm(Jv - Jv2)/norm(Jv2))

    # Check that Newton is converging
    my_step(app, u, 0., 1.1; verbose=true)
    heatmap(u[:, :, 1])
end
# test()

function main(;nx=128, nt=400, tstop=40., len=64., νx=0.2, νy=0.02, A=1., B=3.)
    app = SimApp(nx; νx=νx, νy=νy, len=(len, len), A=A, B=B)
    u = my_init(app, 0.)
    t = 0.
    Δt = tstop / nt
    anim = @animate for i in 1:nt
        println("t = $t")
        my_step(app, u, t, t + Δt)
        heatmap(u[:, :, 1]; clim=(0., maximum(u[:, :, 1])))
        t += Δt
    end

    gif(anim, "brusselator.gif", fps=30)

    return app
end
# main();