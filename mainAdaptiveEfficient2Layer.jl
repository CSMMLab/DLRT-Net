using LinearAlgebra
using PyPlot
using DelimitedFiles

function ReLU(x)
    n = length(x);
    y = zeros(n,1)
    for i = 1:n
        if x[i] > 0
            y[i] = x[i];
            #println("here")
        end
    end
    return y;
end

function dReLU(x)
    n = length(x);
    y = zeros(n,1)
    for i = 1:n
        if x[i] > 0
            y[i] = 1.0;
            #println("here dReLU")
        end
    end
    return y;
end

# loss function
function Loss(z::Array{Float64,1},y::Array{Float64,1})
    return 0.5*norm(z - y,2).^2;
end

function Layer(W::Array{Float64,2},x::Array{Float64,1})
    return ReLU(W*x);
end

function Network(W1::Array{Float64,2},W2::Array{Float64,2},x::Array{Float64,1},y::Array{Float64,1})
    return 0.5*norm(ReLU(W2*ReLU(W1*x)) - y,2).^2;
end

function dNetwork(W1::Array{Float64,2},W2::Array{Float64,2},x::Array{Float64,1},y::Array{Float64,1})
    #an = Wn*znM
    #zn = ReLU(an)
    z0 = x;
    a1 = W1*z0;
    z1 = ReLU(a1);
    
    a2 = W2*z1;
    z2 = ReLU(a2);
    dLoss = (z2 - y);
    dAct1 = dReLU(a1);
    dAct2 = dReLU(a2);
    dweight2 = z1'; # 
    dlayerinput2 = W2;
    dweight1 = z0';
    dLossAct2 = (dLoss.*dAct2)'

    println(size(dLossAct2*z1))
    println(size(dLossAct2*W2))
    println(size(dLossAct2*W2))
    println(size(dAct2))
    println(size((dLossAct2*W2)*dAct1))


    return dLossAct2*dweight2,0;# dLossAct2'*dlayerinput2.*dAct1*dweight1;
end

cStop = 20;

readData = false

if !readData
    N = 10;
    q = 200;
    M = 100;

    W1 = rand(Float64, (q, M));
    W1 ./= norm(W1)
    W2 = rand(Float64, (N, q));
    W2 ./= norm(W2)
    #writedlm("x.txt", x)
    #writedlm("y.txt", y)
    x = rand(M).-0.5
    y = rand(N).-0.5
    writedlm("W1.txt", W1)
    writedlm("W2.txt", W2)
    writedlm("x1.txt", x)
    writedlm("y1.txt", y)
else
    x = vec(readdlm("x.txt"))
    y = vec(readdlm("y.txt"))
    W = readdlm("W.txt")
    N = length(y)
    M = length(x)
end

alpha = 10^-1;
eps = 1e-5;

sdHistory = [ Network(W1,W2,x,y)];

counter = 0;

println("Network SD initial: ", Network(W1,W2,x,y))
# steepest descent
while Network(W1,W2,x,y) > eps && counter <= cStop
    global counter;
    W .= W - alpha*dNetwork(W1,W2,x,y);
    push!(sdHistory, Network(W1,W2,x,y))
    #println("W = ",W)
    println("Network = ",Network(W1,W2,x,y))
    #println("dNetwork = ",dNetwork(W,x,y))
    #println(norm(W))
    counter += 1;
end
#=
############# DLR #############
r = 10;
rMaxTotal = 100;
tol = 1e-2;
epsAdapt = 0.1;
# Low-rank approx of init data:
U,S,V = svd(Wsave); 
    
# rank-r truncation:
U = U[:,1:r]; 
V = V[:,1:r];
S = Diagonal(S);
S = S[1:r, 1:r]; 

K = zeros(N,r);
L = zeros(M,r);

DLRHistory = [Network(U*S*V',x,y)];

# unconventional integrator
counter = 0;
println("Network DLRA initial: ",Network(U*S*V',x,y))
while true
    #println("counter ",counter)
    global counter,U,S,V,r,alpha,tol,epsAdapt;

    ###### K-step ######
    K = U*S;

    K = K .- alpha*dNetworkK(K,V'x,y);

    UNew,STmp = qr([K U]); # optimize bei choosing XFull, SFull
    UNew = UNew[:, 1:2*r]; 

    MUp = UNew' * U;

    ###### L-step ######
    L = V*S';

    L = L .- alpha*dNetworkL(U,L'*x,y);
            
    VNew,STmp = qr([L V]);
    VNew = VNew[:, 1:2*r]; 

    NUp = VNew' * V;
    V = VNew;
    U = UNew;

    ################## S-step ##################
    S = MUp*S*(NUp')

    S .= S .- alpha.*dNetworkS(U,S,V'x,y);

    ################## truncate ##################

    # Compute singular values of S1 and decide how to truncate:
    U2,D,V2 = svd(S);
    rmax = -1;
    S .= zeros(size(S));


    tmp = 0.0;
    tol = epsAdapt*norm(D);
    
    rmax = Int(floor(size(D,1)/2));
    
    for j=1:2*rmax
        tmp = sqrt(sum(D[j:2*rmax]).^2);
        if(tmp<tol)
            rmax = j;
            break;
        end
    end
    
    rmax = min(rmax,rMaxTotal);
    rmax = max(rmax,2);

    for l = 1:rmax
        S[l,l] = D[l];
    end

    # if 2*r was actually not enough move to highest possible rank
    if rmax == -1
        rmax = rMaxTotal;
    end

    # update solution with new rank
    UNew = U*U2;
    VNew = V*V2;

    # update solution with new rank
    S = S[1:rmax,1:rmax];
    U = UNew[:,1:rmax];
    V = VNew[:,1:rmax];

    # update rank
    r = rmax;

    net = Network(U,S,V'*x,y);
    push!(DLRHistory, net)
    println("residual: ",net)
    println("rank: ",r)
    counter +=1;
    Network(U,S,V'*x,y) > eps && counter <= cStop || break
end
=#
fig, ax = subplots()
ax[:plot](collect(1:length(sdHistory)),sdHistory, "k-", linewidth=2, label="sd", alpha=0.6)
#ax[:plot](collect(1:length(DLRHistory)),DLRHistory, "r--", linewidth=2, label="DLR", alpha=0.6)
ax[:legend](loc="upper right")
ax.set_yscale("log")
ax.tick_params("both",labelsize=20) 
show()