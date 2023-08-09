using DataStructures, PrettyTables
using Symbolics, NLsolve

include("build_function_sundials.jl")

struct ButcherTable
    A::AbstractMatrix
    b::AbstractVector
    c::AbstractVector
    s::Int
end
ButcherTable(A, b) = ButcherTable(A, b, A*ones(length(b)), length(b))
ButcherTable(A, b, c) = ButcherTable(A, b, c, length(b))
function ButcherTable(A::Vector{<:Real}, b::Vector{<:Real}, c::Vector{<:Real}, s::Integer)
    return ButcherTable(reshape(A, s, s), b, c, s)
end
function Base.show(io::IO, bt::ButcherTable)
    print(io, "ButcherTable: \n")
    # pretty table
    b = [1.0; bt.b]
    data = [bt.c bt.A; b']
    pretty_table(
        data;
        show_header=false,
        body_hlines=[bt.s],
        vlines=[:begin, 1, :end]
    )
end

forest = OrderedDict(
    # 2nd order
    :{τ} => 1,
    # 3rd order
    :{τ, τ} => 2,
    :{{τ}}  => 3,
    # 4th order
    :{τ, τ, τ} => 4,
    :{τ, {τ}}  => 5,
    :{{τ, τ}}  => 6,
    :{{{τ}}}   => 7,
    # 5th order
    :{τ, τ, τ, τ} => 8,
    :{τ, τ, {τ}}  => 9,
    :{{τ}, {τ}}   => 10,
    :{τ, {τ, τ}}  => 11,
    :{{τ, τ, τ}}  => 12,
    :{τ, {{τ}}}   => 13,
    :{{τ, {τ}}}   => 14,
    :{{{τ, τ}}}   => 15,
    :{{{{τ}}}}    => 16
)

elem_weights = [
    # 2nd order
    (A, b, c) -> b' * c,
    # 3rd order
    (A, b, c) -> b' * c.^2,
    (A, b, c) -> b' * A * c,
    # 4th order
    (A, b, c) -> b' * c.^3,
    (A, b, c) -> b' * (c .* (A*c)),
    (A, b, c) -> b' * A * c.^2,
    (A, b, c) -> b' * A * A * c,
    # 5th order
    (A, b, c) -> b' * c.^4,
    (A, b, c) -> b' * (c .* (c .* (A*c))),
    (A, b, c) -> b' * (A * c).^2,
    (A, b, c) -> b' * (c .* (A*c.^2)),
    (A, b, c) -> b' * A * c.^3,
    (A, b, c) -> b' * (c .* (A*A*c)),
    (A, b, c) -> b' * A * (c .* (A*c)),
    (A, b, c) -> b' * A * A * c.^2,
    (A, b, c) -> b' * A * A * A * c
]

rhs_classical = [
    # 2nd order
    1/2,
    # 3rd order
    1/3, 1/6,
    # 4th order
    1/4, 1/8, 1/12, 1/24,
    # 5th order
    1/5, 1/10, 1/20, 1/15, 1/20, 1/30, 1/40, 1/60, 1/120
]
num_conditions = [0, 1, 3, 7, 16]

ψ(B::ButcherTable, tree::Expr) = elem_weights[forest[tree]](B.A, B.b, B.c)

@variables θ[1:15]

function gen_lhs_func(B::ButcherTable, order::Integer)
    num_conds = num_conditions[order]
    fA, fb, fc = [eval(build_function(syms, collect(θ[1:num_conds]); checkbounds=true)[1]) for syms ∈ (B.A, B.b, B.c)]
    conds_sym = [expand(ψ(B, t)) for t ∈ forest.keys[1:num_conds]]
    conds_jac = Symbolics.jacobian(conds_sym, θ[1:num_conds])
    f = eval(build_function(conds_sym, collect(θ[1:num_conds]); checkbounds=true)[2])
    J = eval(build_function(conds_jac, collect(θ[1:num_conds]); checkbounds=true)[2])
    function fill(θ::AbstractVector)
        @assert length(θ) == num_conds
        ButcherTable(fA(θ), fb(θ), fc(θ), B.s)
    end
    return f, J, fill
end

function solve_order_conditions(f!::Function, J!::Function, fill::Function, order::Integer, rhs::Vector{<:Real};
                                guess=zeros(num_conditions[order]), method=:trust_region, info=false)
    @assert length(rhs) == num_conditions[order]
    function g!(G, θ)
        f!(G, θ)
        G .-= rhs
    end
    result = nlsolve(g!, J!, guess; method=method, ftol=1e-12, xtol=1e-12, iterations=10000)
    if info
        @info result
    elseif !result.f_converged
        @warn "Order conditions solver not converged"
        @info "Ψ: $(rhs')"
        @info result
        result = nlsolve(g!, J!, guess; method=:newton, ftol=1e-12, xtol=1e-12, iterations=1000)
    end
    fill(result.zero)
end

function gen_c_lhs_func(B::ButcherTable, order::Integer, name::AbstractString; filename::AbstractString=name, write::Bool=true, guess=zeros(num_conditions[order]))
    rhsnames = [:th]
    num_conds = num_conditions[order]
    # fill butcher table
    fA = build_function(
        B.A, collect(θ[1:num_conds]);
        target=SunTarget(),
        rowmajor=true,
        fname=name * "_btable_A",
        lhsname=:A, rhsnames=rhsnames
    )
    fb = build_function(
        B.b, collect(θ[1:num_conds]);
        target=SunTarget(),
        fname=name * "_btable_b",
        lhsname=:b, rhsnames=rhsnames
    )
    fc = build_function(
        B.c, collect(θ[1:num_conds]);
        target=SunTarget(),
        fname=name * "_btable_c",
        lhsname=:c, rhsnames=rhsnames
    )
    # compute order conditions
    @variables rhs[1:num_conds]
    conds_sym = [expand(ψ(B, t) - rhs[i]) for (i, t) ∈ enumerate(forest.keys[1:num_conds])]
    conds_jac = Symbolics.jacobian(conds_sym, θ[1:num_conds])
    func = build_function(
        conds_sym, collect(θ[1:num_conds]), collect(rhs[1:num_conds]);
        target=SunTarget(), header=write,
        fname=name * "_res",
        lhsname=:phi, rhsnames=(:th, :rhs)
    )
    func_j = build_function(
        conds_jac, collect(θ[1:num_conds]);
        target=SunTarget(),
        fname=name * "_jac",
        lhsname=:phi_J, rhsnames=rhsnames
    )
    f_guess = build_function(
        [Num(g) for g ∈ guess]; target=SunTarget(),
        fname=name * "_guess",
        lhsname=:guess
    )
    behavior = write ? "w" : "a"
    open("$filename.c", behavior) do file
        println(file, "/* $name */\n")
        println(file, func)
        println(file, func_j)
        println(file, fA)
        println(file, fb)
        println(file, fc)
        println(file, f_guess)
    end
end

beuler = ButcherTable([1.0], [1.0], [1.0])
sdirk212 = ButcherTable([1. 0.; -1. 1.], [1 / 2, 1 / 2], [1., 0.])
sdirk212_emb = [1., 0.]

#sdirk33
x = 0.4358665215
sdirk33 = ButcherTable([x 0.0 0.0; (1-x)/2 x 0; (-3.0x^2/2 + 4.0x - 1/4) (3x^2/2 - 5.0x + 5/4) x], [(-3.0x^2/2 + 4.0x - 1/4), (3.0x^2/2 - 5.0x + 5/4), x])

#esdirk423
diag = 1767732205903 / 4055673282236
A3 = zeros(4, 4)
A3[2, 1] = diag
A3[2, 2] = diag
A3[3, 1] = 2746238789719 / 10658868560708
A3[3, 2] = -640167445237 / 6845629431997
A3[3, 3] = diag
A3[4, 1] = 1471266399579 / 7840856788654
A3[4, 2] = -4482444167858 / 7529755066697
A3[4, 3] = 11266239266428 / 11593286722821
A3[4, 4] = diag
esdirk3 = ButcherTable(A3, A3[4, :], [0, 2diag, 3 / 5, 1])
esdirk3_emb = [2756255671327/12835298489170, -10771552573575/22201958757719, 9247589265047/10645013368117, 2193209047091/5459859503100]

#sdirk534
A4 = [
        1/4       0        0       0     0;
        1/2       1/4      0       0     0;
        17/50    -1/25     1/4     0     0;
        371/1360 -137/2720 15/544  1/4   0;
        25/24    -49/48    125/16 -85/12 1/4
]
sdirk534 = ButcherTable(A4, A4[5, :])
sdirk534_emb = [59/48, -17/96, 225/32, -85/12, 0.]


# θ methods

# filename = "cfuncs/arkode_xbraid_theta_methods"
filename = "../../../../sundials/src/arkode/xbraid/arkode_xbraid_theta_methods"

# can approximate up to 2nd order
θesdirk2 = ButcherTable([0.0 0.0; 1-θ[1] θ[1]], [1-θ[1], θ[1]])
θesdirk2_lhs!, θesdirk2_J!, θesdirk2_fill = gen_lhs_func(θesdirk2, 2)
θesdirk2_guess = [1/2]
gen_c_lhs_func(θesdirk2, 2, "theta_esdirk2"; filename=filename, write=true, guess=θesdirk2_guess)
#=
    0 │     0   0 
    1 │ 1 - θ   θ 
──────┼───────────
    1 │ 1 - θ   θ 
=#

# can approximate up to 2nd order
θsdirk2 = ButcherTable([θ[1] 0.0; 1-θ[1] θ[1]], [1-θ[1], θ[1]])
θsdirk2_lhs!, θsdirk2_J!, θsdirk2_fill = gen_lhs_func(θsdirk2, 2)
θsdirk2_guess = [1-√2/2]
gen_c_lhs_func(θsdirk2, 2, "theta_sdirk2"; filename=filename, write=false, guess=θsdirk2_guess)
#=
    θ │     θ   0 
    1 │ 1 - θ   θ 
──────┼───────────
    1 │ 1 - θ   θ 
=#

# can approximate up to 3rd order
θesdirk3 = ButcherTable([0. 0. 0.; θ[2]-θ[1] θ[1] 0; θ[3] (1.0-θ[3]-θ[1]) θ[1]], [θ[3], 1.0-θ[3]-θ[1], θ[1]])
θesdirk3_lhs!, θesdirk3_J!, θesdirk3_fill = gen_lhs_func(θesdirk3, 3)
θesdirk3_guess = [0.21132486540518713, 0.4226497308103742, 0.1056624327025936]
gen_c_lhs_func(θesdirk3, 3, "theta_esdirk3"; filename=filename, write=false, guess=θesdirk3_guess)
#=
   0  │      0            0   0
   θ₂ │ θ₂ - θ₁           θ₁  0
   1  │      θ₃  1 - θ₃ - θ₁  θ₁
──────┼──────────────────────────
   2  │      θ₃  1 - θ₃ - θ₁  θ₁
=#

θsdirk3 = ButcherTable([θ[1] 0.0 0.0; θ[2]-θ[1] θ[1] 0; (1.0-θ[3]-θ[1]) θ[3] θ[1]], [1.0-θ[3]-θ[1], θ[3], θ[1]])
θsdirk3_lhs!, θsdirk3_J!, θsdirk3_fill = gen_lhs_func(θsdirk3, 3)
θsdirk3_guess = [0.4358665215, 0.71793326075, -0.6443631706532353]
gen_c_lhs_func(θsdirk3, 3, "theta_sdirk3"; filename=filename, write=false, guess=θsdirk3_guess)
#=
   θ₁ │          θ₁  0   0
   θ₂ │     θ₂ - θ₁  θ₁  0
   1  │ 1 - θ₃ - θ₁  θ₃  θ₁
──────┼──────────────────────────
   2  │ 1 - θ₃ - θ₁  θ₃  θ₁
=#

# can approximate up to 4th order
#=
   0  | 0              0     0    0    0
   θ₁ | θ₁/2           θ₁/2  0    0    0
   c₃ | c₃-θ₂-θ₁       θ₂    θ₁   0    0
   c₄ | c₄-θ₃-θ₄-θ₁    θ₃    θ₄   θ₁   0
   1  | 1-θ₅-θ₆-θ₇-θ₁  θ₅    θ₆   θ₇   θ₁
──────┼──────────────────────────────────
   4  | 1-θ₅-θ₆-θ₇-θ₁  θ₅    θ₆   θ₇   θ₁
=#
c3 = 12329209447232/21762223217049
c4 = 2190473621641/2291448330983
# c3 = 1308256777188/2690004194437
# c4 = 2026389075477/2726940318254
θqesdirk4 = ButcherTable(
    [0.0                      0.0    0.0   0.0   0.0;
     θ[1]/2                   θ[1]/2 0.0   0.0   0.0;
     c3-θ[2]-θ[1]             θ[2]   θ[1]  0.0   0.0;
     c4-θ[3]-θ[4]-θ[1]        θ[3]   θ[4]  θ[1]  0.0;
     1.0-θ[5]-θ[6]-θ[7]-θ[1]  θ[5]   θ[6]  θ[7]  θ[1]],
    [1.0-θ[5]-θ[6]-θ[7]-θ[1], θ[5],  θ[6], θ[7], θ[1]]
)
θqesdirk4_lhs!, θqesdirk4_J!, θqesdirk4_fill = gen_lhs_func(θqesdirk4, 4)
θqesdirk4_guess = [0.5065274202451187, -0.1264307159555053, 146.3451690077889, -305.0457869546525, -0.31902898451398665, 0.6494281724470768, 0.001108203953233309]
gen_c_lhs_func(θqesdirk4, 4, "theta_qesdirk4"; filename=filename, write=false, guess=θqesdirk4_guess)

# c2, c3, and c4 coefficients from Hairer and Wanner (1991) pg. 107
#=
  θ₁    | θ₁            0  0  0  0
  3/4   | 3/4-θ₁        θ₁ 0  0  0
  11/20 | 11/20-θ₁-θ₂   θ₂ θ₁ 0  0
  1/2   | 1/2-θ₁-θ₃-θ₄  θ₃ θ₄ θ₁ 0
  1     | 1-θ₁-θ₅-θ₆-θ₇ θ₅ θ₆ θ₇ θ₁
────────┼─────────────────────────────
  1     | 1-θ₁-θ₅-θ₆-θ₇ θ₅ θ₆ θ₇ θ₁
=#
θsdirk4 = ButcherTable(
    [θ[1]                     0    0     0     0;
     3/4-θ[1]               θ[1]   0     0     0;
     11/20-θ[1]-θ[2]        θ[2]   θ[1]  0     0;
     1/2-θ[1]-θ[3]-θ[4]     θ[3]   θ[4]  θ[1]  0;
     1-θ[5]-θ[6]-θ[7]-θ[1]  θ[5]   θ[6]  θ[7]  θ[1]],
    [1-θ[5]-θ[6]-θ[7]-θ[1], θ[5],  θ[6], θ[7], θ[1]]
)
θsdirk4_lhs!, θsdirk4_J!, θsdirk4_fill = gen_lhs_func(θsdirk4, 4)
θsdirk4_guess = [1/4, -1/25, -137/2720, 15/544, -49/48, 125/16, -85/12]
gen_c_lhs_func(θsdirk4, 4, "theta_sdirk4"; filename=filename, write=false, guess=θsdirk4_guess)

#=
  1/4   | 1/4           0  0    0   0
  θ₁    | θ₁-1/4        1/4 0   0   0
  11/20 | 11/20-1/4-θ₂   θ₂  1/4 0   0
  1/2   | 1/2-1/4-θ₃-θ₄  θ₃  θ₄  1/4 0
  1     | 1-1/4-θ₅-θ₆-θ₇ θ₅  θ₆  θ₇  1/4
────────┼─────────────────────────────
  1     | 1-1/4-θ₅-θ₆-θ₇ θ₅  θ₆  θ₇  1/4
=#
# θsdirk4 = ButcherTable(
#     [1/4                   0     0     0     0;
#      θ[1]-1/4              1/4   0     0     0;
#      11/20-1/4-θ[2]        θ[2]  1/4   0     0;
#      1/2-1/4-θ[3]-θ[4]     θ[3]  θ[4]  1/4   0;
#      1-1/4-θ[5]-θ[6]-θ[7]  θ[5]  θ[6]  θ[7]  1/4],
#     [1-1/4-θ[5]-θ[6]-θ[7], θ[5], θ[6], θ[7], 1/4]
# )
# θsdirk4_lhs!, θsdirk4_J!, θsdirk4_fill = gen_lhs_func(θsdirk4, 4)
# θsdirk4_guess = [3/4, -1/25, -137/2720, 15/544, -49/48, 125/16, -85/12]

# θ Lobatto IIIC (4th order)
θLIIIC = ButcherTable(
    [-θ[1]-θ[2] θ[1] θ[2];
     θ[3]  θ[4] θ[5];
     1-θ[6]-θ[7] θ[6] θ[7]],
     [1-θ[6]-θ[7], θ[6], θ[7]]
)
θLIIIC_lhs!, θLIIIC_J!, θLIIIC_fill = gen_lhs_func(θLIIIC, 4)
θLIIIC_guess = [1/6, 1/6, 1/6, 5/12, -1/12, 1/6, 1/6]