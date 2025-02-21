using AnnuliOrthogonalPolynomials, ClassicalOrthogonalPolynomials, InfiniteArrays, LazyBandedMatrices, Test
using AnnuliOrthogonalPolynomials: ColumnInterlace


@testset "ColumnInterlace" begin
    X = ColumnInterlace(jacobimatrix.(Jacobi.(0,1:∞)), (ℵ₀,ℵ₀), (1,1))
    Xₙ = X[Block.(Base.oneto(5)), Block.(Base.oneto(5))]
    @test Xₙ isa ColumnInterlace
    Cₙ = DiagTrav(zeros(5,5))
    Cₙ[1:9] = 1:9
    @test Xₙ*Cₙ ≈ X[Block.(1:5), Block.(1:5)] * Vector(Cₙ)

    for j = 1:5
        @test (Xₙ*Cₙ).array[:,j] ≈ X.ops[j][1:5,1:5] * Cₙ.array[:,j]
    end
    C = DiagTrav(zeros(∞,∞))
    C[1:9] = 1:9
    @test (X*C)[Block.(1:5)] ≈ Xₙ*Cₙ
end