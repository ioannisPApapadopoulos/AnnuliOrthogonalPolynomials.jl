module AnnuliOrthogonalPolynomials
using BlockArrays, BandedMatrices, BlockBandedMatrices, ClassicalOrthogonalPolynomials, ContinuumArrays, DomainSets, FastTransforms, FillArrays,
    HarmonicOrthogonalPolynomials, LazyArrays, LinearAlgebra, LazyBandedMatrices,
    MultivariateOrthogonalPolynomials, QuasiArrays, SemiclassicalOrthogonalPolynomials, StaticArrays, ArrayLayouts, InfiniteArrays


import ArrayLayouts: MemoryLayout, sublayout, sub_materialize
import Base: in, axes, getindex, broadcasted, tail, +, -, *, /, \, convert, OneTo, show, summary, ==, oneto, diff, copy, view, BroadcastStyle
import BandedMatrices: _BandedMatrix
import BlockArrays: block, blockindex, _BlockedUnitRange, blockcolsupport, BlockSlice
import BlockBandedMatrices: blockbandwidths, subblockbandwidths, AbstractBandedBlockBandedLayout, AbstractBandedBlockBandedMatrix
import ClassicalOrthogonalPolynomials: checkpoints, ShuffledR2HC, TransformFactorization, ldiv, paddeddata, jacobimatrix, orthogonalityweight, SetindexInterlace
import ContinuumArrays: Weight, weight, grid, ℵ₁, ℵ₀, unweighted, plan_transform, @simplify
import HarmonicOrthogonalPolynomials: BivariateOrthogonalPolynomial, MultivariateOrthogonalPolynomial, Plan
import MultivariateOrthogonalPolynomials: BlockOneTo, ModalInterlace, laplacian, ModalTrav
import SemiclassicalOrthogonalPolynomials: divdiff, HalfWeighted, SemiclassicalJacobiFamily
import LazyBandedMatrices: AbstractLazyBandedBlockBandedLayout
import LazyArrays: resizedata!
using InfiniteArrays: InfiniteCardinal

export Block, SVector, CircleCoordinate, ZernikeAnnulus, ComplexZernikeAnnulus, Laplacian, JacobiDiskSlice

include("columninterlace.jl")
include("annulus.jl")
include("diskslice.jl")

end # module AnnuliOrthogonalPolynomials
