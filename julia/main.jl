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
    return Loss(Layer(W,x),y);
end

function dNetwork(W::Array{Float64,2},x::Array{Float64,1},y::Array{Float64,1})
    return (ReLU(W*x)-y).*dReLU(W*x)*x';
end

function ReLU(x)
    n = length(x);
    y = zeros(n)
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

cStop = 2;

readData = false

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

sdHistory = [Network(W,x,y)];

counter = 0;

println("Network SD initial: ",Network(W,x,y))
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
while Network(U*S*V',x,y) > eps && counter <= cStop
    global counter;

    gradient = dNetwork(U*S*V',x,y);

    ###### K-step ######
    K .= U*S;

    K .= K .- alpha*gradient*V;
    println(size(K))

    UNew,STmp = qr(K); # optimize bei choosing XFull, SFull
    UNew = UNew[:, 1:r]; 

    NUp = UNew' * U;

    ###### L-step ######
    L .= V*S';

    L .= L .- alpha*(U'*gradient)';
    println("L")
    println(size(L))

    VNew,STmp = qr(L);
    VNew = VNew[:, 1:r]; 
    println("vnew")
    println(size(VNew))
    println("STmp")
    println(size(STmp))
    MUp = VNew' * V;
    V .= VNew;
    U .= UNew;

    ################## S-step ##################
    S .= NUp*S*(MUp')

    S .= S .- alpha.*U'*dNetwork(U*S*V',x,y)*V;
    push!(DLRHistory, Network(U*S*V',x,y))
    println("residual: ",Network(U*S*V',x,y))
    counter +=1;
end

fig, ax = subplots()
ax[:plot](collect(1:length(sdHistory)),sdHistory, "k-", linewidth=2, label="sd", alpha=0.6)
ax[:plot](collect(1:length(DLRHistory)),DLRHistory, "r--", linewidth=2, label="DLR", alpha=0.6)
ax[:legend](loc="upper right")
ax.set_yscale("log")
ax.tick_params("both",labelsize=20) 
show()