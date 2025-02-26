annulus(ρ::T) where T = UnitDisk{T}() \ (ρ*UnitDisk{T}())
ClassicalOrthogonalPolynomials.checkpoints(d::DomainSets.SetdiffDomain{SVector{2, T}, Tuple{DomainSets.EuclideanUnitBall{2, T, :closed}, DomainSets.GenericBall{SVector{2, T}, :closed, T}}}) where T = [SVector{2,T}(cos(0.1),sin(0.1)), SVector{2,T}(cos(0.2),sin(0.2))]

"""
    AnnulusWeight(a, b)

is a quasi-vector representing `(r^2-ρ)^a * (1-r^2)^b`
"""
struct AnnulusWeight{T} <: Weight{T}
    ρ::T
    a::T
    b::T
end


AnnulusWeight{T}(ρ) where T = AnnulusWeight{T}(ρ, zero(T), zero(T))
AnnulusWeight(ρ) = AnnulusWeight{Float64}(ρ)

copy(w::AnnulusWeight) = w

axes(w::AnnulusWeight{T}) where T = (Inclusion(annulus(w.ρ)),)

==(w::AnnulusWeight, v::AnnulusWeight) = w.a == v.a && w.b == v.b && w.ρ == v.ρ

function getindex(w::AnnulusWeight, xy::StaticVector{2})
    r = norm(xy)
    (r^2- w.ρ^2)^w.a * (1-r^2)^w.b
end

abstract type AbstractZernikeAnnulus{T} <: BivariateOrthogonalPolynomial{T} end
"""
    ZernikeAnnulus(ρ, a, b)

is a quasi-matrix orthogonal `(r^2 - ρ^2)^a * (1-r^2)^b`.
"""
struct ZernikeAnnulus{T} <: AbstractZernikeAnnulus{T}
    ρ::T
    a::T
    b::T
    ZernikeAnnulus{T}(ρ::T, a::T, b::T) where T = new{T}(ρ, a, b)
end

"""
    ComplexZernikeAnnulus(ρ, a, b)

is a complex-valued quasi-matrix orthogonal `(r^2 - ρ^2)^a * (1-r^2)^b`.
"""
struct ComplexZernikeAnnulus{T} <: AbstractZernikeAnnulus{Complex{T}}
    ρ::T
    a::T
    b::T
    ComplexZernikeAnnulus{T}(ρ::T, a::T, b::T) where T = new{T}(ρ, a, b)
end


for Typ in (:ZernikeAnnulus, :ComplexZernikeAnnulus)
    @eval begin
        $Typ{T}(ρ, a, b) where T = $Typ{T}(convert(T,ρ), convert(T,a), convert(T,b))
        $Typ(ρ::R, a::T, b::V) where {R,T,V} = $Typ{float(promote_type(R,T,V))}(ρ, a, b)
        $Typ{T}(ρ) where T = $Typ{T}(ρ, zero(ρ), zero(ρ))
        $Typ(ρ) = $Typ(ρ, zero(ρ), zero(ρ))

        ==(w::$Typ, v::$Typ) = w.ρ == v.ρ && w.a == v.a && w.b == v.b
    end
end



axes(P::AbstractZernikeAnnulus{T}) where T = (Inclusion(annulus(P.ρ)),blockedrange(oneto(∞)))
copy(A::AbstractZernikeAnnulus) = A

orthogonalityweight(Z::AbstractZernikeAnnulus) = AnnulusWeight(Z.ρ, Z.a, Z.b)

zernikeannulusr(ρ, ℓ, m, a, b, r::T) where T = r^m * SemiclassicalJacobi{T}(inv(1-ρ^2),b,a,m)[(r^2 - 1)/(ρ^2 - 1), (ℓ-m) ÷ 2 + 1]
function zernikeannulusz(ρ, ℓ, ms, a, b, rθ::RadialCoordinate{T}) where T
    r,θ = rθ.r,rθ.θ
    m = abs(ms)
    zernikeannulusr(ρ, ℓ, m, a, b, r) * (signbit(ms) ? sin(m*θ) : cos(m*θ))
end

function complexzernikeannulusz(ρ, ℓ, ms, a, b, rθ::RadialCoordinate{T}) where T
    r,θ = rθ.r,rθ.θ
    m = abs(ms)
    zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ)
end


for func in (:zernikeannulusz, :complexzernikeannulusz)
    @eval begin
        $func(ρ, ℓ, ms, a, b, xy::StaticVector{2}) = $func(ρ, ℓ, ms, a, b, RadialCoordinate(xy))
        $func(ρ, ℓ, ms, b, xy::StaticVector{2}) = $func(ρ, ℓ, ms, zero(b), b, xy)
        $func(ρ, ℓ, ms, xy::StaticVector{2,T}) where T = $func(ρ, ℓ, ms, zero(T), xy)
    end
end

function getindex(Z::ZernikeAnnulus{T}, rθ::RadialCoordinate, B::BlockIndex{1}) where T
    ℓ = Int(block(B))-1
    k = blockindex(B)
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k)
    zernikeannulusz(Z.ρ, ℓ, (isodd(k+ℓ) ? 1 : -1) * m, Z.a, Z.b, rθ)
end


function getindex(Z::ComplexZernikeAnnulus{T}, rθ::RadialCoordinate, B::BlockIndex{1}) where T
    ℓ = Int(block(B))-1
    k = blockindex(B)
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k)
    complexzernikeannulusz(Z.ρ, ℓ, (isodd(k+ℓ) ? 1 : -1) * m, Z.a, Z.b, rθ)
end


getindex(Z::AbstractZernikeAnnulus, xy::StaticVector{2}, B::BlockIndex{1}) = Z[RadialCoordinate(xy), B]
getindex(Z::AbstractZernikeAnnulus, xy::StaticVector{2}, B::Block{1}) = [Z[xy, B[j]] for j=1:Int(B)]
getindex(Z::AbstractZernikeAnnulus, xy::StaticVector{2}, JR::BlockOneTo) = mortar([Z[xy,Block(J)] for J = 1:Int(JR[end])])



function \(A::AbstractZernikeAnnulus, B::AbstractZernikeAnnulus)
    TV = promote_type(eltype(A),eltype(B))
    A.a == B.a && A.b == B.b && return Eye{TV}(∞)
    t = inv(1-A.ρ^2)
    ModalInterlace{TV}((SemiclassicalJacobi{real(TV)}.(t,A.b,A.a,0:∞) .\ SemiclassicalJacobi{real(TV)}.(t,B.b,B.a,0:∞)), (ℵ₀,ℵ₀), (0,2*Int(max(A.b-B.b,A.a-B.a))))
end

function \(A::ZernikeAnnulus{T}, B::Weighted{V,ZernikeAnnulus{V}}) where {T,V}
    TV = promote_type(T,V)
    (A.a == B.P.a == A.b == B.P.b == 0 && A.ρ == B.P.ρ) && return Eye{TV}(∞)
    @assert A.a == A.b == 1
    @assert B.P.a == B.P.b == 1
    @assert A.ρ == B.P.ρ

    ρ = convert(TV, A.ρ); t=inv(one(TV)-ρ^2)

    # L₁ = Weighted.(SemiclassicalJacobi{real(TV)}.(t,zero(TV),zero(TV),zero(TV):∞)) .\ Weighted.(SemiclassicalJacobi{real(TV)}.(t,one(TV),one(TV),zero(TV):∞))
    # L₂ = SemiclassicalJacobi{real(TV)}.(t,one(TV),one(TV),zero(TV):∞) .\ SemiclassicalJacobi{real(TV)}.(t,zero(TV),zero(TV),zero(TV):∞)

    Q₀₀ = SemiclassicalJacobi{real(TV)}.(t,0,0,0:∞)
    Q₁₁ = SemiclassicalJacobi{real(TV)}.(t,1,1,0:∞)

    L₁ = Weighted.(Q₀₀) .\ Weighted.(Q₁₁)
    L₂ = Q₁₁ .\ Q₀₀

    # L = (one(TV)-ρ^2)^2 .* (L₂ .* L₁)
    # Workaround for broken lazy multiplication
    L = BroadcastVector{AbstractMatrix{TV}}((L2, L1) -> (one(TV)-ρ^2)^2 .* ApplyArray(*,L2,L1), L₂, L₁)
    ModalInterlace{TV}(L, (ℵ₀,ℵ₀), (4, 4))
end


###
# Laplacian
###


function laplacian(W::Weighted{<:Any,<:ZernikeAnnulus})
    P = W.P
    @assert P.a == P.b == 1
    ρ = P.ρ; t = inv(1-ρ^2)
    T = eltype(P)
    Ps = SemiclassicalJacobi{T}.(t,1,1,0:∞)
    Δs = BroadcastVector{AbstractMatrix{T}}((C,B,A) -> 4t*(1-ρ^2)^2*divdiff(HalfWeighted{:c}(C), HalfWeighted{:c}(B))*divdiff(HalfWeighted{:ab}(B), HalfWeighted{:ab}(A)), Ps, SemiclassicalJacobi.(t,0,0,1:∞), Ps)
    P * ModalInterlace(Δs, (ℵ₀,ℵ₀), (2,2))
end

function laplacian(P::ZernikeAnnulus)
    ρ,a,b = P.ρ,P.a,P.b
    t = inv(1-ρ^2)
    T = eltype(P)
    Ps = SemiclassicalJacobi.(t,b,a,0:∞)
    Δs = BroadcastVector{AbstractMatrix{T}}((C,B,A) -> 4t*divdiff(HalfWeighted{:c}(C), HalfWeighted{:c}(B))*divdiff(B, A), SemiclassicalJacobi.(t,b+2,a+2,0:∞), SemiclassicalJacobi.(t,b+1,a+1,1:∞), Ps)
    ZernikeAnnulus(ρ,a+2,b+2) * ModalInterlace(Δs, (ℵ₀,ℵ₀), (-2,6))
end

###
# Transforms
###

function grid(S::ZernikeAnnulus{T}, B::Block{1}) where T
    N = Int(B) ÷ 2 + 1 # polynomial degree
    M = 4N-3
    ρ = parent(S).ρ

    # The radial grid:
    r = [first([begin t = (N-n-one(T)/2)/(2N); ct = sinpi(t); st = cospi(t); sqrt(ct^2+ρ^2*st^2) end]) for n in 0:N-1]

    # The angular grid:
    θ = (0:M-1)*convert(T,2)/M
    RadialCoordinate.(r, convert(T,π)*θ')
end

_angle(rθ::RadialCoordinate) = rθ.θ

function plotgrid(S::ZernikeAnnulus{T}, B::Block{1}) where T
    N = Int(B) ÷ 2 + 1  # polynomial degree
    g = grid(S[:,Block.(OneTo(2N))]) # double sampling
    θ = [map(_angle,g[1,:]); 0]
    ρ = S.ρ + eps(T) # Without the eps, it sporadically throws a BoundsError.
    [permutedims(RadialCoordinate.(1,θ)); g g[:,1]; permutedims(RadialCoordinate.(ρ,θ))]
end
plotgrid(wS::Weighted{<:Any,<:ZernikeAnnulus}, B::Block{1}) = plotgrid(unweighted(wS), B)

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{ZernikeAnnulus, AbstractVector}}, x) where T
    Z,c = u.args
    CS = blockcolsupport(c)
    N = Int(last(CS)) ÷ 2 + 1 # polynomial degree
    F = ZernikeAnnulusITransform{T}(2N, Z.a, Z.b, 0, Z.ρ) # 0 should be Z.c if implemented.
    C = F * c[Block.(OneTo(2N))] # transform to grid
    [permutedims(u[x[1,:]]); # evaluate on edge of disk
     C C[:,1];
     permutedims(u[x[end,:]])] # evaluate at edge of annulus and repeat
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{Weighted{<:Any,<:ZernikeAnnulus}, AbstractVector}}, x) where T
    U = plotvalues(unweighted(u), x)
    w = weight(u.args[1])
    w[x] .* U
end


# FastTransforms uses orthonormalized annulus OPs so we need to correct the normalization
# Gives a vector with the weights of each m-mode.
function normalize_mmodes(w::AnnulusWeight{T}) where T
    (ρ,a,b) = convert(T,w.ρ), convert(T,w.a), convert(T,w.b)
    t = inv(one(T)-ρ^2)
    # contribution from non-normalized semiclassical Jacobi
    jw = sum.(SemiclassicalJacobiWeight{T}.(t,b,a,0:∞))
    # π comes from contribution of Harmonic polynomials and t^(a+b+one(T)) from the change of variables
    m₀ =  sqrt( convert(T,π) / t^(a+b+one(T)) ) * sqrt(jw[1])
    vcat([m₀], sqrt.( convert(T,π) ./ (2 * t.^( (a+b+one(T)) .+ one(T):∞))).* sqrt.(jw[2:end]))
end

# Creates vector correctly interlacing the denormalization
# constants for each m-mode.
function denormalize_annulus(A::AbstractVector, a, b, c, ρ, analysis=true)
    l = length(A)
    bl = [findblockindex(blockedrange(oneto(∞)), j) for j in 1:l]
    ℓ = [bl[j].I[1]-1 for j in 1:l] # degree
    k = [bl[j].α[1] for j in 1:l]   # index of degree
    m = [iseven(ℓ[j]) ? k[j]-isodd(k[j]) : k[j]-iseven(k[j]) for j in 1:l] # m-mode
    s = (-1).^Int.((ℓ .- m) ./ 2) # denormalization includes negatives (-1)^(degree - m)/2
    w = AnnulusWeight(ρ, a, b)
    constants = normalize_mmodes(w)[1:l] # m-mode constants
    d = [inv(constants[mm+1]*ss) for (mm, ss) in zip(m, s)] # multiply by relevant (-1)
    analysis && return d.*A # multiply vector by denormalization if analysis
    A ./ d # divide vector by denormalization if synthesis
end

# # FastTransforms uses orthonormalized annulus OPs so we need to correct the normalization
# # Gives a vector with the weights of each m-mode.
# function normalize_mmodes(w::AnnulusWeight{T}, l::Int) where T
#     (ρ,a,b) = convert(T,w.ρ), convert(T,w.a), convert(T,w.b)
#     t = inv(one(T)-ρ^2)
#     # contribution from non-normalized semiclassical Jacobi
#     jw = sum.(SemiclassicalJacobiWeight.(t,b,a,0:(l-1)))
#     # π comes from contribution of Harmonic polynomials and t^(a+b+one(T)) from the change of variables
#     m₀ =  sqrt( convert(T,π) / t^(a+b+one(T)) ) * sqrt(jw[1])
#     vcat([m₀], sqrt.( convert(T,π) ./ (2 * t.^( (a+b+2*one(T)):(convert(T,l)+a+b)))) .* sqrt.(jw[2:end]))
# end

# # Creates vector correctly interlacing the denormalization
# # constants for each m-mode.
# function denormalize_annulus(A::AbstractVector, a, b, c, ρ, analysis=true)
#     supp = Int(last(blockcolsupport(A)))
#     l = length(A[Block.(1:supp)])
#     bl = [findblockindex(blockedrange(oneto(∞)), j) for j in 1:l]
#     ℓ = [bl[j].I[1]-1 for j in 1:l] # degree
#     k = [bl[j].α[1] for j in 1:l]   # index of degree
#     m = [iseven(ℓ[j]) ? k[j]-isodd(k[j]) : k[j]-iseven(k[j]) for j in 1:l] # m-mode
#     s = (-1).^Int.((ℓ .- m) ./ 2) # denormalization includes negatives (-1)^(degree - m)/2
#     w = AnnulusWeight(ρ, a, b)
#     constants = normalize_mmodes(w, l) # m-mode constants
#     d = [inv(constants[mm+1]*ss) for (mm, ss) in zip(m, s)] # multiply by relevant (-1)
#     analysis && return [d.*A[1:l];A[l+1:end]] # multiply vector by denormalization if analysis
#     [A[1:l]./d;A[l+1:end]]  # divide vector by denormalization if synthesis
# end

struct ZernikeAnnulusTransform{T} <: Plan{T}
    N::Int
    ann2cxf::FastTransforms.FTPlan{T,2,FastTransforms.ANNULUS}
    analysis::FastTransforms.FTPlan{T,2,FastTransforms.ANNULUSANALYSIS}
    a::T
    b::T
    c::T
    ρ::T
end

struct ZernikeAnnulusITransform{T} <: Plan{T}
    N::Int
    ann2cxf::FastTransforms.FTPlan{T,2,FastTransforms.ANNULUS}
    synthesis::FastTransforms.FTPlan{T,2,FastTransforms.ANNULUSSYNTHESIS}
    a::T
    b::T
    c::T
    ρ::T
end

function ZernikeAnnulusTransform{T}(N::Int, a::Number, b::Number, c::Number, ρ::Number) where T<:Real
    @assert c == 0 # Remove when/if the weight r^(2c) is added.
    Ñ = N ÷ 2 + 1
    ZernikeAnnulusTransform{T}(N, plan_ann2cxf(T, Ñ, a, b, c, ρ), plan_annulus_analysis(T, Ñ, 4Ñ-3, ρ), a, b, c, ρ)
end
function ZernikeAnnulusITransform{T}(N::Int, a::Number, b::Number, c::Number, ρ::Number) where T<:Real
    @assert c == 0 # Remove when/if the weight r^c is added.
    Ñ = N ÷ 2 + 1
    ZernikeAnnulusITransform{T}(N, plan_ann2cxf(T, Ñ, a, b, c, ρ), plan_annulus_synthesis(T, Ñ, 4Ñ-3, ρ), a, b, c, ρ)
end

*(P::ZernikeAnnulusTransform{T}, f::AbstractArray) where T = P * convert(Matrix{T}, f)
*(P::ZernikeAnnulusTransform{T}, f::Matrix{T}) where T = denormalize_annulus(ModalTrav(P.ann2cxf \ (P.analysis * f)), P.a, P.b, P.c, P.ρ)
*(P::ZernikeAnnulusITransform, f::AbstractVector) = P.synthesis * (P.ann2cxf * ModalTrav(denormalize_annulus(f, P.a, P.b, P.c, P.ρ, false)).matrix)

plan_transform(S::ZernikeAnnulus{T}, (B,)::Tuple{Block{1}}, dims=1:1) where T = ZernikeAnnulusTransform{T}(Int(B), S.a, S.b, zero(T), S.ρ)