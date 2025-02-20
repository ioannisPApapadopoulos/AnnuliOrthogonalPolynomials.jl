using AnnuliOrthogonalPolynomials, StaticArrays, QuadGK, Test

@testset "DiskSlice" begin
    P = JacobiDiskSlice(0.0)

    @test P[SVector(0.1,0.2),1] ≈ 1

    @testset "orthogonality" begin
        @test quadgk(x -> quadgk(y -> P[SVector(x,y),1], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1] ≈ π/2
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),2], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),3], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),4], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),5], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),3]P[SVector(x,y),2], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),4]P[SVector(x,y),2], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),5]P[SVector(x,y),2], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),4]P[SVector(x,y),3], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),5]P[SVector(x,y),3], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
        @test abs(quadgk(x -> quadgk(y -> P[SVector(x,y),5]P[SVector(x,y),4], -sqrt(1-x^2), sqrt(1-x^2), atol=1e-8)[1], 0, 1, atol=1e-8)[1]) ≤ 1E-8
    end

    @testset "Jacobi matrix" begin
        x,y = 𝐱 = SVector(0.1,0.2)
    
        
        α = 0.0
        X_R = (α-1)*jacobimatrix(P.P[1]) + I
        X_C = jacobimatrix(Ultraspherical(P.b+1/2))

        @test x*P.P[1][(x-1)/(P.α-1),1] ≈ X_R.dv[1]*P.P[1][(x-1)/(P.α-1),1] + X_R.ev[1]*P.P[1][(x-1)/(P.α-1),2]
        @test x * P[𝐱,1] ≈ X_R.dv[1]*P[𝐱,1] + X_R.ev[1]*P[𝐱,2]

        for n = 0:5, k=0:n
            X_R = (α-1)*jacobimatrix(P.P[k+1]) + I
            if k < n
                @test x * P[𝐱,Block(n+1)[k+1]] ≈ X_R.ev[n-k]*P[𝐱,Block(n)[k+1]] + X_R.dv[n-k+1]*P[𝐱,Block(n+1)[k+1]] + X_R.ev[n-k+1]*P[𝐱,Block(n+2)[k+1]]
            else # n == k
                @test x * P[𝐱,Block(n+1)[k+1]] ≈ X_R.dv[1]*P[𝐱,Block(n+1)[k+1]] + X_R.ev[1]*P[𝐱,Block(n+2)[k+1]]
            end
        end

        X = jacobimatrix(Val(1), P)
        @test x * P[𝐱,Block.(1:4)]' ≈ P[𝐱,Block.(1:5)]' * X[Block.(1:5),Block.(1:4)]
    end

end
