"""
    DiskSlice(α)

represents α ≤ x ≤ 1, y^2 ≤ 1-x^2.
"""
struct DiskSlice{T} <: Domain{T}
    α::T
end


function ClassicalOrthogonalPolynomials.checkpoints(d::DiskSlice)
    α = d.α
    x = α + (1-α)/3
    [SVector(x, sqrt(1-x^2)/7)]
end

"""
    DiskSliceWeight(α, a, b)

is a quasi-vector representing `(x-α)^a * (1-x^2-y^2)^b`
"""
struct DiskSliceWeight{T} <: Weight{T}
    α::T
    a::T
    b::T
end

copy(w::DiskSliceWeight) = w

axes(w::DiskSliceWeight{T}) where T = (Inclusion(DiskSlice(w.α)),)

==(w::DiskSliceWeight, v::DiskSliceWeight) = w.a == v.a && w.b == v.b && w.α == v.α

function getindex(w::DiskSliceWeight, 𝐱::StaticVector{2})
    x,y = 𝐱
    (x-w.α)^w.a * (1-x^2-y^2)^w.b
end

"""
    JacobiDiskSlice(α, a, b)

is a quasi-matrix orthogonal to `(x-α)^a * (1-x^2-y^2)^b`.
"""
struct JacobiDiskSlice{T} <: BivariateOrthogonalPolynomial{T} 
    α::T
    a::T
    b::T
    P::SemiclassicalJacobiFamily{T}
    JacobiDiskSlice{T}(α::T, a::T, b::T) where T = new{T}(α, a, b, SemiclassicalJacobiFamily(2/(1-α), (b+one(T)/2):∞, a, (b+one(T)/2):∞))
end

JacobiDiskSlice{T}(α, a, b) where T = JacobiDiskSlice{T}(convert(T,α), convert(T,a), convert(T,b))
JacobiDiskSlice(α::R, a::T, b::V) where {R,T,V} = JacobiDiskSlice{float(promote_type(R,T,V))}(α, a, b)
JacobiDiskSlice{T}(α) where T = JacobiDiskSlice{T}(α, zero(α), zero(α))
JacobiDiskSlice(α) = JacobiDiskSlice(α, zero(α), zero(α))

convert(::Type{AbstractQuasiMatrix{T}}, P::JacobiDiskSlice) where T = JacobiDiskSlice{T}(P.α, P.a, P.b)

==(w::JacobiDiskSlice, v::JacobiDiskSlice) = w.α == v.α && w.a == v.a && w.b == v.b

axes(P::JacobiDiskSlice{T}) where T = (Inclusion(DiskSlice(P.α)),blockedrange(oneto(∞)))
copy(A::JacobiDiskSlice) = A

orthogonalityweight(P::JacobiDiskSlice) = DiskSliceWeight(P.α, P.a, P.b)


function getindex(P::JacobiDiskSlice{T}, 𝐱::StaticVector{2}, B::BlockIndex{1}) where T
    n,k = Int(block(B)), blockindex(B)
    x,y = 𝐱
    ρ = sqrt(1-x^2)
    P.P[k][(x-1)/(P.α-1),n-k+1] * ρ^(k-1) * ultrasphericalc(k-1, P.b+one(T)/2, y/ρ)
end
getindex(P::JacobiDiskSlice, 𝐱::StaticVector{2}, B::Block{1}) = [P[𝐱, B[j]] for j=1:Int(B)]
getindex(P::JacobiDiskSlice, 𝐱::StaticVector{2}, JR::BlockOneTo) = mortar([P[𝐱,Block(J)] for J = 1:Int(JR[end])])


###
# jacobimatrix
###

jacobimatrix(::Val{1}, P::JacobiDiskSlice) = ColumnInterlace(BroadcastVector{AbstractMatrix{Float64}}((α,Pₖ) -> (α-1)*jacobimatrix(Pₖ) + I, P.α, P.P), (ℵ₀,ℵ₀), (1,1))


###
# raising
####

@simplify function \(Q::JacobiDiskSlice, P::JacobiDiskSlice)
    T = promote_type(eltype(Q), eltype(P))
    @assert Q.a == P.a + 1 && Q.b == P.b
    ColumnInterlace(BroadcastVector{AbstractMatrix{T}}(\, Q.P, P.P), (ℵ₀,ℵ₀), (0,1))
end