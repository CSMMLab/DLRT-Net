using LinearAlgebra
using PyPlot
using DelimitedFiles

# loss function
function Loss(z::Array{Float64,1},y::Array{Float64,1})
    return 0.5*norm(z - y,2).^2;
end

function Layer(W::Array{Float64,2},x::Array{Float64,1})
    return ReLU(W*x);
end

function Network(W::Array{Float64,2},x::Array{Float64,1},y::Array{Float64,1})
    return  0.5*norm(Layer(W,x) - y,2).^2;
end

function Network(U::Array{Float64,2},S::Array{Float64,2},VTx::Array{Float64,1},y::Array{Float64,1})
    return 0.5*norm(ReLU(U*S*VTx) - y,2).^2;
end

function dNetwork(W::Array{Float64,2},x::Array{Float64,1},y::Array{Float64,1})
    return (ReLU(W*x)-y).*dReLU(W*x)*x';
end

function dNetworkK(K::Array{Float64,2},VTx::Array{Float64,1},y::Array{Float64,1})
    return (ReLU(K*VTx)-y).*dReLU(K*VTx)*(VTx');
end

function dNetworkL(U::Array{Float64,2},LTx::Array{Float64,1},y::Array{Float64,1})
    return ((ReLU(U*LTx)-y).*dReLU(U*LTx)*x')'*U;
end

function dNetworkS(U::Array{Float64,2},S::Array{Float64,2},VTx::Array{Float64,1},y::Array{Float64,1})
    return U'*((ReLU(U*S*VTx)-y).*dReLU(U*S*VTx))*(VTx');
end

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
    y = zeros(n)
    for i = 1:n
        if x[i] > 0
            y[i] = 1.0;
            #println("here dReLU")
        end
    end
    return y;
end

cStop = 2000;

readData = true

if !readData
    N = 1000;
    M = 1000;
    x = rand(M).-0.5;
    x ./= norm(x)*10;
    y = rand(N).-0.5;
    y ./= norm(y);
    W = rand(Float64, (N, M));
    W ./= norm(W)
    writedlm("x.txt", x)
    writedlm("y.txt", y)
    writedlm("W.txt", W)
else
    x = vec(readdlm("x.txt"))
    y = vec(readdlm("y.txt"))
    W = readdlm("W.txt")
    N = length(y)
    M = length(x)
end

Wsave = deepcopy(W);

# Low-rank approx of init data:
WTarget = rand(Float64, (N, M));
WTarget ./= norm(WTarget)*1e-3;
U,S,V = svd(WTarget); 
rt = 15;
# rank-r truncation:
U = U[:,1:rt]; 
V = V[:,1:rt];
S = Diagonal(S);
S = S[1:rt, 1:rt]; 

WTargetlow = U*S*V';
y = WTargetlow*x;

alpha = 10^-1;
eps = 1e-5;

sdHistory = [ Network(W,x,y)];

counter = 0;

println("Network SD initial: ", Network(W,x,y))
# steepest descent
while Network(W,x,y) > eps && counter <= cStop
    global counter;
    W .= W - alpha*dNetwork(W,x,y);
    push!(sdHistory, Network(W,x,y))
    #println("W = ",W)
    println("Network = ",Network(W,x,y))
    #println("dNetwork = ",dNetwork(W,x,y))
    #println(norm(W))
    counter += 1;
end

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
    println("counter: ",counter)
    counter +=1;
    Network(U,S,V'*x,y) > eps && counter <= cStop || break
end

fig, ax = subplots()
ax[:plot](collect(1:length(sdHistory)),sdHistory, "k-", linewidth=2, label="sd", alpha=0.6)
ax[:plot](collect(1:length(DLRHistory)),DLRHistory, "r--", linewidth=2, label="DLR", alpha=0.6)
ax[:legend](loc="upper right")
ax.set_yscale("log")
ax.tick_params("both",labelsize=20) 
show()