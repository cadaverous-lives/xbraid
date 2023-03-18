using LinearAlgebra, PreallocationTools, ForwardDiff, DiffResults
using MPI, Interpolations, IterativeSolvers
using SparseArrays, ToeplitzMatrices
using FFTW
using Plots
# theme(:dracula)

include("XBraid.jl")
using .XBraid

MPI.Init()
comm = MPI.COMM_WORLD

struct BurgerApp
    x::Vector{Float64}
    κ::Frequencies{Float64}
    cf::Integer
    useTheta::Bool
    # preallocated caches that support ForwardDiff:
    x_d::DiffCache
    y_d::DiffCache
    # storage for solution values
    solution
    lyap_vecs
    lyap_exps
    times
    # pre-planned fourier transform
    P̂
end
function BurgerApp(x::Vector{Float64}, κ::Frequencies{Float64}, cf::Integer, useTheta::Bool, x_d::DiffCache, y_d::DiffCache)
    BurgerApp(x, κ, cf, useTheta, x_d, y_d, [], [], [], [], plan_fft(x))
end

# system parameters (globally scoped)
const lengthScale = 64.
const nₓ = 1024
const Δx = lengthScale / nₓ

function stencil_to_circulant(stencil::Vector{T}, nx::Integer) where T
    @assert isodd(length(stencil)) "stencil should have odd number of entries"
    stencilWidth = length(stencil)
    center = ceil(Int, length(stencil)/2)
    columnView = reverse(stencil)
    v = [columnView[center:end]; zeros(T, nx-stencilWidth); columnView[1:center-1]]
    return Circulant(v)
end

function euler_mat(Δt::Float64; μ::Float64=.00001)
    # diffusion
    # A = spdiagm(-1 => ones(nₓ-1), 0 => -2*ones(nₓ), 1 => ones(nₓ-1))
    # A[1, end] = 1
    # A[end, 1] = 1
    # A = sparse(I, (nₓ, nₓ)) - Δt/Δx^2 * μ * A

    # KS operator
    ∇⁴ = [1, -4, 6, -4, 1]
    ∇² = [0, 1, -2, 1, 0]
    I  = [0, 0, 1, 0, 0]
    stencil = @. I - Δt*(-1/Δx^4 * ∇⁴ - 1/Δx^2 * ∇²)
    # stencil = @. I - Δt(-μ/Δx^4 * ∇⁴)
    stencil_to_circulant(stencil, nₓ)
end

function semi_lagrangian!(burger::BurgerApp, y, Δt)
    x_back = get_tmp(burger.x_d, y)
    y_intp = get_tmp(burger.y_d, y)

    x_back .= burger.x .- Δt * y
    itp = interpolate(y, BSpline(Linear(Periodic())))
    sitp = scale(itp, range(0., lengthScale - Δx, nₓ))
    extp = extrapolate(sitp, Periodic())
    # y_intp .= interp_periodic.(x_back, Ref(y))
    y_intp .= extp.(x_back)
    y .= y_intp
    return y
end

function diffuse_beuler!(burger::BurgerApp, y, Δt::Float64; init_guess=nothing)
    y_tmp = get_tmp(burger.y_d, y)
    y_tmp .= y

    if init_guess !== nothing
        y .= init_guess
    end

    cg!(y, euler_mat(Δt), y_tmp)
    return y
end

function diffuse_fft!(burger::BurgerApp, y, Δt::Float64; init_guess=nothing, μ=1e-4)
    P̂ = burger.P̂
    κ = burger.κ
    # y .= real(P̂ \ (exp.(Δt*(κ.^2 - κ.^4))))
    ŷ = P̂ * y
    # @. ŷ *= exp(-μ * Δt * κ^2) # standard diffusion
    @. ŷ *= exp(Δt*(κ^2 - κ^4)) # KS-equation operator
    y .= real(P̂ \ ŷ)
    return y
end

# user routines:
function my_init(burger::BurgerApp, t::Float64)
    u = similar(burger.x)
    u .= sin.(2π/lengthScale * burger.x)
    if t == 0.
        u .*= randn(nₓ)
    end
    # @. u = exp(-(burger.x - lengthScale/2)^2)
    return u
end

function my_basis_init(burger::BurgerApp, t::Float64, k::Int32)
    x = burger.x
    ψ = similar(x)
    if k % 2 == 0
        @. ψ = cos(k/2*x)
    else
        @. ψ = sin((k+1)/2*x)
    end
    return ψ
end

function base_step!(burger::BurgerApp, u, Δt::Float64; init_guess=nothing)
    # diffuse_fft!(burger, u, Δt/2)
    semi_lagrangian!(burger, u, Δt)
    diffuse_fft!(burger, u, Δt)
    # diffuse_beuler!(burger, u, Δt; init_guess=init_guess)
end

function my_step!(
    burger::BurgerApp, status::Ptr{Cvoid}, 
    u, ustop, tstart::Float64, tstop::Float64
)
    Δt = tstop - tstart
    level = XBraid.status_GetLevel(status)
    if !burger.useTheta || level == 0
        base_step!(burger, u, Δt; init_guess=ustop)
    else
        # richardson based θ method
        m = burger.cf^level
        θ = 2(m - 1)/m
        # θ = 2
        u_sub = deepcopy(u)
        base_step!(burger, u_sub, Δt/2; init_guess=(ustop .- u)./2)
        base_step!(burger, u_sub, Δt/2; init_guess=ustop)
        base_step!(burger, u, Δt; init_guess=u_sub)
        @. u = θ*u_sub + (1-θ)*u
    end
    return u
end

function my_step!(burger::BurgerApp, status::Ptr{Cvoid}, u, ustop, tstart::Float64, tstop::Float64, Ψ)
    rank = length(Ψ)
    Ψ_new = reduce(hcat, Ψ)
    # perturb(r) = base_step!(burger, u + r' * Ψ, Δt)
    perturb(r) = my_step!(burger, status, u + r' * Ψ, ustop, tstart, tstop)

    result = DiffResults.DiffResult(u, Ψ_new)
    result = ForwardDiff.jacobian!(result, perturb, zeros(rank))
    for i in eachindex(Ψ)
        Ψ[i] .= Ψ_new[:, i]
    end
end

function my_sum!(burger, a, x, b, y)
    @. y = a*x + b*y
end

function my_access(burger, status, u)
    XBraid.status_GetWrapperTest(status) && return
    ti = XBraid.status_GetTIndex(status)
    push!(burger.solution, deepcopy(u))
    push!(burger.times, ti)
    if XBraid.status_GetDeltaRank(status) > 0
        Ψ = XBraid.status_GetBasisVectors(status)
        λ = XBraid.status_GetLocalLyapExponents(status)
        push!(burger.lyap_vecs, deepcopy(reduce(hcat, Ψ)))
        push!(burger.lyap_exps, deepcopy(λ))
    end
end

my_norm(burger, u) = LinearAlgebra.norm2(u)
my_innerprod(burger, u, v) = u' * v

# test user routines:
function test()
    x_new = zeros(nₓ)
    y_new = zeros(nₓ)
    x = Array(range(0., lengthScale-Δx, nₓ))
    κ = 2π/lengthScale .* fftfreq(nₓ, nₓ)
    burger = BurgerApp(x, κ, 4, false, DiffCache(x_new), DiffCache(y_new));
    test_app = XBraid.BraidApp(
        burger, comm, comm,
        my_step!, my_init,
        my_sum!, my_norm, my_access,
        my_basis_init, my_innerprod)

    open("drive-burgers.test.out", "w") do file
        cfile = Libc.FILE(file)
        XBraid.testBuf(test_app, 0.0, cfile)
        XBraid.testSpatialNorm(test_app, 0.0, cfile)
        XBraid.testDelta(test_app, 0.0, 0.1, 3, cfile)
    end
    # plot(burger.solution[1])
    # plot!(burger.lyap_vecs[1][1])
end

function main(;tstop=20., ntime=128, deltaRank=0, useTheta=false, ml=1, cf=4, saveGif=false)
    x_new = zeros(nₓ)
    y_new = zeros(nₓ)
    x = Array(range(0., lengthScale-Δx, nₓ))
    κ = 2π/lengthScale .* fftfreq(nₓ, nₓ)
    burger = BurgerApp(x, κ, 4, useTheta, DiffCache(x_new), DiffCache(y_new));

    tstart = 0.0
    core = XBraid.Init(
        comm, comm, tstart, tstop, ntime,
        my_step!, my_init, my_sum!, my_norm, my_access; app=burger
    )

    if deltaRank > 0
        XBraid.SetDeltaCorrection(core, deltaRank, my_basis_init, my_innerprod)
        XBraid.SetLyapunovEstimation(core; exponents=true)
    end

    # println("Wrapper test:")
    # @time test()

    XBraid.SetMaxIter(core, 30)
    XBraid.SetMaxLevels(core, ml)
    XBraid.SetCFactor(core, -1, cf)
    XBraid.SetAccessLevel(core, 1)
    XBraid.SetNRelax(core, -1, 1)
    XBraid.SetAbsTol(core, 1e-6)

    XBraid.SetTimings(core, 2)
    XBraid.Drive(core; warmup=true)
    XBraid.PrintTimers(core)

    if deltaRank > 0
        exponents = sum(burger.lyap_exps)
        exponents = MPI.Allreduce(exponents, (+), comm)
        exponents ./= tstop
        if MPI.Comm_rank(comm) == 0
            println("exponents: ", exponents)
        end
    end

    if saveGif
        for (ti, u) in zip(burger.times, burger.solution)
            plt = plot(burger.x, u; ylim=(-1.5, 1.5))
            savefig(plt, "burgers_gif/$(lpad(ti, 6, "0")).png")
        end
        MPI.Barrier(comm)

        if MPI.Comm_rank(comm) == 0
            println("animating...")
            fnames = [lpad(i, 6, "0") for i in 0:ntime]
            anim = Animation("burgers_gif", fnames)
            Plots.buildanimation(anim, "burgers_gif/anim.gif", fps=30)
        end
    end

    return burger, XBraid.GetRNorms(core)
end

# test()
main();
nothing