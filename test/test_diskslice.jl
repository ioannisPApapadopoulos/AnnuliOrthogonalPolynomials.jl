using AnnuliOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, Test
using QuadGK

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

        @testset "Jacobi-X" begin
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

        @testset "Jacobi-Y" begin
            n,k = 5,2
            α,a,b = P.α,P.a,P.b
            ρ = sqrt(1-x^2)

            Rₖ = P.P[k+2] \ P.P[k+1]
            Lₖ = Weighted(P.P[k]) \ Weighted(P.P[k+1])

            @test P.P[k+1][(x-1)/(α-1),n+1-(k+1)+1]  ≈ P.P[k+2][(x-1)/(α-1),n+1-(k+1)-1] * Rₖ[n+1-(k+1)-1,n+1-(k+1)+1] + P.P[k+2][(x-1)/(α-1),n+1-(k+1)] * Rₖ[n+1-(k+1),n+1-(k+1)+1] + P.P[k+2][(x-1)/(α-1),n+1-(k+1)+1] * Rₖ[n+1-(k+1)+1,n+1-(k+1)+1]

            t = 2/(1-α)
            τ = (x-1)/(α-1)
            @test τ * (t-τ) ≈ (1 - x^2)/(1-α)^2 

            @test (1-x^2)*P.P[k+1][τ,n-k+1] ≈ (1-α)^2 * τ * (t-τ) * P.P[k+1][τ,n-k+1] ≈ (1-α)^2 * (P.P[k][τ,n-k+1] * Lₖ[n-k+1,n-k+1] + P.P[k][τ,n-k+2] * Lₖ[n-k+2,n-k+1] + P.P[k][τ,n-k+3] * Lₖ[n-k+3,n-k+1])
            
            

            @testset "derivation" begin
                @test y*P[𝐱, Block(n+1)[k+1]] ≈ y * P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^k * ultrasphericalc(k,b+1/2,y/ρ) ≈
                            P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k+1) * y/ρ * ultrasphericalc(k,b+1/2,y/ρ) ≈
                            P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k+1) * (X_C[k,k+1]ultrasphericalc(k-1,b+1/2,y/ρ) + X_C[k+2,k+1]ultrasphericalc(k+1,b+1/2,y/ρ)) ≈
                            X_C[k,k+1]P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k+1) * ultrasphericalc(k-1,b+1/2,y/ρ) + X_C[k+2,k+1]P.P[k+1][(x-1)/(α-1),n+1-(k+1)+1] * ρ^(k+1) * ultrasphericalc(k+1,b+1/2,y/ρ) ≈
                            X_C[k,k+1]P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k+1) * ultrasphericalc(k-1,b+1/2,y/ρ) + X_C[k+2,k+1]*(P.P[k+2][(x-1)/(α-1),n+1-(k+1)-1] * Rₖ[n+1-(k+1)-1,n+1-(k+1)+1] + P.P[k+2][(x-1)/(α-1),n+1-(k+1)] * Rₖ[n+1-(k+1),n+1-(k+1)+1] + P.P[k+2][(x-1)/(α-1),n+1-(k+1)+1] * Rₖ[n+1-(k+1)+1,n+1-(k+1)+1]) * ρ^(k+1) * ultrasphericalc(k+1,b+1/2,y/ρ) ≈
                            X_C[k,k+1]*(1-x^2)*P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k-1) * ultrasphericalc(k-1,b+1/2,y/ρ) + X_C[k+2,k+1]Rₖ[n+1-(k+1)-1,n+1-(k+1)+1]P[𝐱,Block(n)[k+2]] +  X_C[k+2,k+1]Rₖ[n+1-(k+1),n+1-(k+1)+1]P[𝐱,Block(n+1)[k+2]] +  X_C[k+2,k+1]Rₖ[n+1-(k+1)+1,n+1-(k+1)+1]P[𝐱,Block(n+2)[k+2]] ≈
                            (1-α)^2 * X_C[k,k+1]Lₖ[n-k+1,n-k+1] * P[𝐱,Block(n)[k]] + X_C[k+2,k+1]Rₖ[n+1-(k+1)-1,n+1-(k+1)+1]P[𝐱,Block(n)[k+2]]  +
                            (1-α)^2 * X_C[k,k+1]Lₖ[n-k+2,n-k+1] *  P[𝐱,Block(n+1)[k]] + X_C[k+2,k+1]Rₖ[n+1-(k+1),n+1-(k+1)+1]P[𝐱,Block(n+1)[k+2]] +
                            (1-α)^2 * X_C[k,k+1]Lₖ[n-k+3,n-k+1] * P[𝐱,Block(n+2)[k]] + X_C[k+2,k+1]Rₖ[n+1-(k+1)+1,n+1-(k+1)+1]P[𝐱,Block(n+2)[k+2]]
            end
        end
    end

    @testset "derivative" begin
        x,y = 𝐱 = SVector(0.1,0.2)
        h = sqrt(eps())
        (P[SVector(x+h,y),Block(n+1)[k+1]]-P[𝐱,Block(n+1)[k+1]])/h
        
    end
end
