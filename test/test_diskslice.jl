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
        X_U = jacobimatrix(Ultraspherical(P.b+1/2))

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

            @test y*P[𝐱, Block(n+1)[k+1]] ≈
            (1-α)^2 * X_U[k,k+1]Lₖ[n-k+1,n-k+1] * P[𝐱,Block(n)[k]] + X_U[k+2,k+1]Rₖ[n-k-1,n-k+1]P[𝐱,Block(n)[k+2]]  +
            (1-α)^2 * X_U[k,k+1]Lₖ[n-k+2,n-k+1] *  P[𝐱,Block(n+1)[k]] + X_U[k+2,k+1]Rₖ[n-k,n-k+1]P[𝐱,Block(n+1)[k+2]] +
            (1-α)^2 * X_U[k,k+1]Lₖ[n-k+3,n-k+1] * P[𝐱,Block(n+2)[k]] + X_U[k+2,k+1]Rₖ[n-k+1,n-k+1]P[𝐱,Block(n+2)[k+2]]
            
            
            @testset "derivation" begin
                @test P.P[k+1][(x-1)/(α-1),n+1-(k+1)+1]  ≈ P.P[k+2][(x-1)/(α-1),n+1-(k+1)-1] * Rₖ[n+1-(k+1)-1,n+1-(k+1)+1] + P.P[k+2][(x-1)/(α-1),n+1-(k+1)] * Rₖ[n+1-(k+1),n+1-(k+1)+1] + P.P[k+2][(x-1)/(α-1),n+1-(k+1)+1] * Rₖ[n+1-(k+1)+1,n+1-(k+1)+1]
    
                t = 2/(1-α)
                τ = (x-1)/(α-1)
                @test τ * (t-τ) ≈ (1 - x^2)/(1-α)^2 
    
                @test (1-x^2)*P.P[k+1][τ,n-k+1] ≈ (1-α)^2 * τ * (t-τ) * P.P[k+1][τ,n-k+1] ≈
                    (1-α)^2 * (P.P[k][τ,n-k+1] * Lₖ[n-k+1,n-k+1] + P.P[k][τ,n-k+2] * Lₖ[n-k+2,n-k+1] + P.P[k][τ,n-k+3] * Lₖ[n-k+3,n-k+1])
                @test y*P[𝐱, Block(n+1)[k+1]] ≈ y * P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^k * ultrasphericalc(k,b+1/2,y/ρ) ≈
                            P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k+1) * y/ρ * ultrasphericalc(k,b+1/2,y/ρ) ≈
                            P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k+1) * (X_U[k,k+1]ultrasphericalc(k-1,b+1/2,y/ρ) + X_U[k+2,k+1]ultrasphericalc(k+1,b+1/2,y/ρ)) ≈
                            X_U[k,k+1]P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k+1) * ultrasphericalc(k-1,b+1/2,y/ρ) + X_U[k+2,k+1]P.P[k+1][(x-1)/(α-1),n+1-(k+1)+1] * ρ^(k+1) * ultrasphericalc(k+1,b+1/2,y/ρ) ≈
                            X_U[k,k+1]P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k+1) * ultrasphericalc(k-1,b+1/2,y/ρ) + X_U[k+2,k+1]*(P.P[k+2][(x-1)/(α-1),n+1-(k+1)-1] * Rₖ[n+1-(k+1)-1,n+1-(k+1)+1] + P.P[k+2][(x-1)/(α-1),n+1-(k+1)] * Rₖ[n+1-(k+1),n+1-(k+1)+1] + P.P[k+2][(x-1)/(α-1),n+1-(k+1)+1] * Rₖ[n+1-(k+1)+1,n+1-(k+1)+1]) * ρ^(k+1) * ultrasphericalc(k+1,b+1/2,y/ρ) ≈
                            X_U[k,k+1]*(1-x^2)*P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k-1) * ultrasphericalc(k-1,b+1/2,y/ρ) + X_U[k+2,k+1]Rₖ[n+1-(k+1)-1,n+1-(k+1)+1]P[𝐱,Block(n)[k+2]] +  X_U[k+2,k+1]Rₖ[n+1-(k+1),n+1-(k+1)+1]P[𝐱,Block(n+1)[k+2]] +  X_U[k+2,k+1]Rₖ[n+1-(k+1)+1,n+1-(k+1)+1]P[𝐱,Block(n+2)[k+2]] ≈
                            (1-α)^2 * X_U[k,k+1]Lₖ[n-k+1,n-k+1] * P[𝐱,Block(n)[k]] + X_U[k+2,k+1]Rₖ[n-k-1,n-k+1]P[𝐱,Block(n)[k+2]]  +
                            (1-α)^2 * X_U[k,k+1]Lₖ[n-k+2,n-k+1] *  P[𝐱,Block(n+1)[k]] + X_U[k+2,k+1]Rₖ[n-k,n-k+1]P[𝐱,Block(n+1)[k+2]] +
                            (1-α)^2 * X_U[k,k+1]Lₖ[n-k+3,n-k+1] * P[𝐱,Block(n+2)[k]] + X_U[k+2,k+1]Rₖ[n-k+1,n-k+1]P[𝐱,Block(n+2)[k+2]]
            end
        end
    end

    @testset "Raising" begin
        n,k = 5,2
        x,y = 𝐱 = SVector(0.1,0.2)
        @testset "raise a" begin
            P = JacobiDiskSlice(0)
            Q = JacobiDiskSlice(0, 1, 0)
            R = Q.P[k+1] \ P.P[k+1]
            @test P[𝐱, Block(n+1)[k+1]] ≈ P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^k * ultrasphericalc(k,b+1/2,y/ρ) ≈
                Q.P[k+1][(x-1)/(α-1),n-k+1]R[n-k+1,n-k+1] * ρ^k * ultrasphericalc(k,b+1/2,y/ρ) + Q.P[k+1][(x-1)/(α-1),n-k]R[n-k,n-k+1] * ρ^k * ultrasphericalc(k,b+1/2,y/ρ) ≈
                Q[𝐱, Block(n+1)[k+1]]R[n-k+1,n-k+1] + Q[𝐱, Block(n)[k+1]]R[n-k,n-k+1]

            R = Q\P
            
            @test Q[𝐱,1:10]'R[1:10,1:10] ≈ P[𝐱,1:10]'
        end

        @testset "raise b" begin
            α,a,b = 0.0,0,0
            P = JacobiDiskSlice(α, a, b)
            Q = JacobiDiskSlice(α, a, b+1)
            U = Ultraspherical(b+1/2)
            C = Ultraspherical(b+1/2+1)
            R_U = C\U
            Lₖ = Weighted(Q.P[k-1]) \ Weighted(Q.P[k])
            Rₖ = Q.P[k+1] \ Q.P[k]
            ρ = sqrt(1-x^2)
            τ = (x-1)/(α-1)
            t = Q.P.t
            @test t ≈ 2/(1-α)
            @test (α-1)*τ+1 ≈ x
            @test 1-x^2 ≈ (α-1)^2*τ * (t-τ)
            @test P[𝐱, Block(n+1)[k+1]] ≈ P.P[k+1][τ,n-k+1] * ρ^k * U[y/ρ, k+1] ≈
                P.P[k+1][τ,n-k+1] * ρ^k * C[y/ρ, k-1]*R_U[k-1,k+1] + P.P[k+1][τ,n-k+1] * ρ^k * C[y/ρ, k+1]*R_U[k+1,k+1] ≈
                (α-1)^2*τ * (t-τ) * Q.P[k][τ,n-k+1] * ρ^(k-2) * C[y/ρ, k-1]*R_U[k-1,k+1] + Q.P[k][τ,n-k+1] * ρ^k * C[y/ρ, k+1]*R_U[k+1,k+1] ≈
                (α-1)^2*(Q.P[k-1][τ,n-k+1]*Lₖ[n-k+1,n-k+1] + Q.P[k-1][τ,n-k+2]*Lₖ[n-k+2,n-k+1]+Q.P[k-1][τ,n-k+3]*Lₖ[n-k+3,n-k+1]) * ρ^(k-2) * C[y/ρ, k-1]*R_U[k-1,k+1] + 
                (Q.P[k+1][τ,n-k+1]*Rₖ[n-k+1,n-k+1] + Q.P[k+1][τ,n-k]*Rₖ[n-k,n-k+1]+Q.P[k+1][τ,n-k-1]*Rₖ[n-k-1,n-k+1]) * ρ^k * C[y/ρ, k+1]*R_U[k+1,k+1] ≈
                (α-1)^2*Q[𝐱, Block(n-1)[k-1]]*R_U[k-1,k+1]*Lₖ[n-k+1,n-k+1] + (α-1)^2*Q[𝐱, Block(n)[k-1]]*Lₖ[n-k+2,n-k+1]*R_U[k-1,k+1]+(α-1)^2*Q[𝐱, Block(n+1)[k-1]]*Lₖ[n-k+3,n-k+1]*R_U[k-1,k+1] + 
                Q[𝐱, Block(n+1)[k+1]]*Rₖ[n-k+1,n-k+1]*R_U[k+1,k+1] + Q[𝐱, Block(n)[k+1]]*Rₖ[n-k,n-k+1]R_U[k+1,k+1]+Q[𝐱, Block(n-1)[k+1]]*Rₖ[n-k-1,n-k+1]*R_U[k+1,k+1] ≈
                (α-1)^2*Q[𝐱, Block(n-1)[k-1]]R_U[k-1,k+1]Lₖ[n-k+1,n-k+1] + Q[𝐱, Block(n-1)[k+1]]*Rₖ[n-k-1,n-k+1]*R_U[k+1,k+1] +
                (α-1)^2*Q[𝐱, Block(n)[k-1]]*Lₖ[n-k+2,n-k+1]*R_U[k-1,k+1] + Q[𝐱, Block(n)[k+1]]*Rₖ[n-k,n-k+1]R_U[k+1,k+1] +
                (α-1)^2*Q[𝐱, Block(n+1)[k-1]]*Lₖ[n-k+3,n-k+1]*R_U[k-1,k+1] + Q[𝐱, Block(n+1)[k+1]]*Rₖ[n-k+1,n-k+1]*R_U[k+1,k+1]
        end
    end

    @testset "derivative" begin
        x,y = 𝐱 = SVector(0.1,0.2)
        h = sqrt(eps())
        n,k = 5,2
        
        α,b = P.α,P.b
        U = Ultraspherical(b+1/2)
        C = Ultraspherical(b+1/2+1)

        ρ = sqrt(1-x^2)
        ρ′ = -x/ρ
        D_U = C \ diff(U)

        @testset "d/dx" begin
            @test P[𝐱,Block(n+1)[k+1]] == P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^k * U[y/ρ,k+1]
            Pₓ = 1/(α-1) * diff(P.P[k+1])[(x-1)/(α-1),n-k+1] * ρ^k * U[y/ρ,k+1] +
            k * ρ′ * P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k-1) * U[y/ρ,k+1] -
            y * ρ′/ρ^2 * P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^k * diff(U)[y/ρ,k+1]
            @test (P[SVector(x+h,y),Block(n+1)[k+1]]-P[𝐱,Block(n+1)[k+1]])/h ≈ Pₓ atol=1e-6

            Q = JacobiDiskSlice(0,1,1)
            D = Q.P[k+1] \ diff(P.P[k+1])
            R_U = Ultraspherical(3/2) \ Ultraspherical(1/2)
            D_U = Ultraspherical(3/2) \ diff(Ultraspherical(1/2))
            τ = (x-1)/(α-1)
            @test Pₓ ≈ 1/(α-1) * (Q.P[k+1][τ,n-k-1]*D[n-k-1,n-k+1] + Q.P[k+1][τ,n-k]*D[n-k,n-k+1]) * ρ^k * U[y/ρ,k+1] +
            -x * k * P.P[k+1][τ,n-k+1] * ρ^(k-2) * U[y/ρ,k+1] +
            x * y * P.P[k+1][τ,n-k+1] * ρ^(k-3) * diff(U)[y/ρ,k+1] ≈
            1/(α-1) * Q.P[k+1][τ,n-2-k+1] * ρ^k * C[y/ρ,k+1]D[n-k-1,n-k+1]R_U[k+1,k+1] + 
            (α-1)*τ * (t-τ) * Q.P[k+1][τ,n-k-1] * ρ^(k-2) *C[y/ρ,k-1]D[n-k-1,n-k+1]R_U[k-1,k+1] +
            1/(α-1) * Q.P[k+1][τ,n-k]*D[n-k,n-k+1] * ρ^k * (C[y/ρ,k+1]R_U[k+1,k+1] + C[y/ρ,k-1]R_U[k-1,k+1]) +
            -x * k * P.P[k+1][τ,n-k+1] * ρ^(k-2) * (C[y/ρ,k+1]R_U[k+1,k+1] + C[y/ρ,k-1]R_U[k-1,k+1]) +
            x * y * P.P[k+1][τ,n-k+1] * ρ^(k-3) * C[y/ρ,k]D_U[k,k+1]
        end

        @testset "d/dy" begin
            P_y = P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k-1) * diff(U)[y/ρ,k+1]
            @test (P[SVector(x,y+h),Block(n+1)[k+1]]-P[𝐱,Block(n+1)[k+1]])/h ≈ P_y atol=1e-6
            Q₁ = JacobiDiskSlice(0, 0, 1)
            @test P_y ≈ P.P[k+1][(x-1)/(α-1),n-k+1] * ρ^(k-1) * C[y/ρ,k] * D_C[k,k+1] ≈ Q₁[𝐱, Block(n)[k]]
        end
    end
end
