# using LazyArrays: eltype
# using ForwardDiff: derivative, hessian
# using AlgebraicCurveOrthogonalPolynomials, LinearAlgebra, BlockBandedMatrices, BlockArrays, FillArrays, Test
using AnnuliOrthogonalPolynomials, Test

include("test_annulus.jl")
include("test_diskslice.jl")