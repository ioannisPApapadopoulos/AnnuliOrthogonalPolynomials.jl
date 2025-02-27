"""
    ColumnInterlace(ops, (M,N), (l,u))

interlaces the entries of a vector of banded matrices
acting on different columns of the matrix underling a `DiagTrav`. 
"""
struct ColumnInterlace{T, MMNN<:Tuple} <: AbstractBandedBlockBandedMatrix{T}
    ops
    MN::MMNN
    bandwidths::NTuple{2,Int}
end

ColumnInterlace{T}(ops, MN::NTuple{2,Integer}, bandwidths::NTuple{2,Int}) where T = ColumnInterlace{T,typeof(MN)}(ops, MN, bandwidths)
ColumnInterlace(ops::AbstractVector{<:AbstractMatrix}, MN::NTuple{2,Integer}, bandwidths::NTuple{2,Int}) = ColumnInterlace{eltype(eltype(ops))}(ops, MN, bandwidths)

axes(Z::ColumnInterlace) = blockedrange.(oneto.(Z.MN))

blockbandwidths(R::ColumnInterlace) = R.bandwidths
subblockbandwidths(::ColumnInterlace) = (0,0)
copy(M::ColumnInterlace) = M


function view(R::ColumnInterlace{T}, KJ::Block{2}) where T
    K,J = KJ.n
    dat = Matrix{T}(undef,1,J)
    l,u = blockbandwidths(R)
    if -l ≤ J - K ≤ u
        sh = (J-K)
        for j in 1:min(K,J)
            dat[1,j] = R.ops[j][K-j+1,J-j+1]
        end
    else
        fill!(dat, zero(T))
    end
    _BandedMatrix(dat, K, 0, 0)
end

getindex(R::ColumnInterlace, k::Integer, j::Integer) = R[findblockindex.(axes(R),(k,j))...]

struct ColumnInterlaceLayout <: AbstractBandedBlockBandedLayout end
struct LazyColumnInterlaceLayout <: AbstractLazyBandedBlockBandedLayout end

MemoryLayout(::Type{<:ColumnInterlace}) = ColumnInterlaceLayout()
MemoryLayout(::Type{<:ColumnInterlace{<:Any,NTuple{2,InfiniteCardinal{0}}}}) = LazyColumnInterlaceLayout()
sublayout(::Union{ColumnInterlaceLayout,LazyColumnInterlaceLayout}, ::Type{<:NTuple{2,BlockSlice{<:BlockOneTo}}}) = ColumnInterlaceLayout()


function sub_materialize(::ColumnInterlaceLayout, V::AbstractMatrix{T}) where T
    kr,jr = parentindices(V)
    KR,JR = kr.block,jr.block
    M,N = Int(last(KR)), Int(last(JR))
    R = parent(V)
    ColumnInterlace{T}([layout_getindex(R.ops[k],1:(M-k+1),1:(N-k+1)) for k=1:min(N,M)], (M,N), R.bandwidths)
end

# act like lazy array
BroadcastStyle(::Type{<:ColumnInterlace{<:Any,NTuple{2,InfiniteCardinal{0}}}}) = LazyArrayStyle{2}()

# TODO: overload muladd!
function *(A::ColumnInterlace, b::DiagTrav)
    M = b.array
    ret = zeros(promote_type(eltype(A), eltype(b)), size(M)...)
    m = maximum(colsupport(M))
    n = maximum(rowsupport(M))
    resizedata!(ret, m, n)
    for j = 1:n
        mul!(view(ret,1:m-j+1,j), view(A.ops[j], 1:m-j+1, 1:m-j+1), view(M,1:m-j+1,j))
    end
    DiagTrav(ret)
end


function \(A::ColumnInterlace, b::DiagTrav)
    M = b.matrix
    ret = similar(M, promote_type(eltype(A),eltype(b)))
    for j = 1:size(ret,2)
        ldiv!(@view(ret[1:end-j+1,j]), A.ops[j], @view(M[1:end-j+1,j]))
    end
    DiagTrav(ret)
end
