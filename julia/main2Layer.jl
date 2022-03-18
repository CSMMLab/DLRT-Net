using LinearAlgebra
using PyPlot
using DelimitedFiles

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

    dz1 = dReLU(a1);
    dz2 = dReLU(a2);
    
    return (z2.-y).*dz2*z1',vec(((z2.-y).*dz2)'*W2).*dz1*vec(x)';# dLossAct2'*dlayerinput2.*dAct1*dweight1;
end

cStop = 20;

readData = true

if !readData
    N = 100;
    q = 2000;
    M = 150;

    W1 = rand(Float64, (q, M));
    W1 ./= norm(W1)
    W2 = rand(Float64, (N, q));
    W2 ./= norm(W2)
    #writedlm("x.txt", x)
    #writedlm("y.txt", y)
    x = rand(M).-0.5
    y = rand(N).-0.5

    W1Target = rand(Float64, (q, M));
    W1Target ./= norm(W1Target)*1e-1;
    U,S,V = svd(W1Target); 
    rt = 15;
    # rank-r truncation:
    U = U[:,1:rt]; 
    V = V[:,1:rt];
    S = Diagonal(S);
    S = S[1:rt, 1:rt]; 

    W1Targetlow = U*S*V';

    W2Target = rand(Float64, (N,q));
    W2Target ./= norm(W2Target)*1e-1;
    U,S,V = svd(W2Target); 
    rt = 15;
    # rank-r truncation:
    U = U[:,1:rt]; 
    V = V[:,1:rt];
    S = Diagonal(S);
    S = S[1:rt, 1:rt]; 

    W2Targetlow = U*S*V';

    y = ReLU(W2Target*ReLU(W1Target*x));

    writedlm("W1.txt", W1)
    writedlm("W2.txt", W2)
    writedlm("x1.txt", x)
    writedlm("y1.txt", y)
else
    x = vec(readdlm("x1.txt"))
    y = vec(readdlm("y1.txt"))
    W1 = readdlm("W1.txt")
    W2 = readdlm("W2.txt")
    N = length(y)
    M = length(x)
    q = size(W1,1)
end

W1Save = deepcopy(W1);
W2Save = deepcopy(W2);



alpha = 10^-2;
eps = 1e-5;

sdHistory = [ Network(W1,W2,x,y)];

counter = 0;

println("Network SD initial: ", Network(W1,W2,x,y))
# steepest descent
while Network(W1,W2,x,y) > eps && counter <= cStop
    global counter;
    dW2,dW1 = dNetwork(W1,W2,x,y);
    W1 .= W1 .- alpha*dW1;
    W2 .= W2 .- alpha*dW2;
    push!(sdHistory, Network(W1,W2,x,y))
    #println("W = ",W)
    println("Network = ",Network(W1,W2,x,y))
    #println("dNetwork = ",dNetwork(W,x,y))
    #println(norm(W))
    counter += 1;
end


############# DLR #############
r = 10;
# Low-rank approx of init data:
U1,S1,V1 = svd(W1Save); 
    
# rank-r truncation:
U1 = U1[:,1:r]; 
V1 = V1[:,1:r];
S1 = Diagonal(S1);
S1 = S1[1:r, 1:r]; 

K1 = zeros(q,r);
L1 = zeros(M,r);

U2,S2,V2 = svd(W2Save); 
    
# rank-r truncation:
U2 = U2[:,1:r]; 
V2 = V2[:,1:r];
S2 = Diagonal(S2);
S2 = S2[1:r, 1:r]; 

K2 = zeros(N,r);
L2 = zeros(q,r);

DLRHistory = [Network(U1*S1*V1',U2*S2*V2',x,y)];

# unconventional integrator
counter = 0;
println("Network DLRA initial: ",Network(U1*S1*V1',U2*S2*V2',x,y))
while Network(U1*S1*V1',U2*S2*V2',x,y) > eps && counter <= cStop
    global counter;

    gradient2,gradient1 = dNetwork(U1*S1*V1',U2*S2*V2',x,y);

    ################## K-step W1 ##################
    K1 .= U1*S1;

    K1 .= K1 .- alpha*gradient1*V1;

    U1New,STmp = qr(K1); # optimize bei choosing XFull, SFull
    U1New = U1New[:, 1:r]; 

    M1Up = U1New' * U1;

    ################## L-step W1 ##################
    L1 .= V1*S1';

    L1 .= L1 .- alpha*(U1'*gradient1)';
            
    V1New,STmp = qr(L1);
    V1New = V1New[:, 1:r]; 

    N1Up = V1New' * V1;

    ################## S-step W1 ##################
    S1New = M1Up*S1*(N1Up')

    tmp,gradient1New = dNetwork(U1New*S1New*V1New',U2*S2*V2',x,y)

    S1New .= S1New .- alpha.*U1New'*gradient1New*V1New;

    ################## K-step W2 ##################
    K2 .= U2*S2;

    K2 .= K2 .- alpha*gradient2*V2;

    U2New,STmp = qr(K2); # optimize bei choosing XFull, SFull
    U2New = U2New[:, 1:r]; 

    M2Up = U2New' * U2;

    ################## L-step W1 ##################
    L2 .= V2*S2';

    L2 .= L2 .- alpha*(U2'*gradient2)';
            
    V2New,STmp = qr(L2);
    V2New = V2New[:, 1:r]; 

    N2Up = V2New' * V2;

    ################## S-step W1 ##################
    S2New = M2Up*S2*(N2Up')

    gradient2,tmp = dNetwork(U1*S1*V1',U2New*S2New*V2New',x,y)

    S2New .= S2New .- alpha.*U2New'*gradient2*V2New;

    V1 .= V1New;
    U1 .= U1New;
    S1 .= S1New;
    V2 .= V2New;
    U2 .= U2New;
    S2 .= S2New;
    push!(DLRHistory, Network(U1*S1*V1',U2*S2*V2',x,y))
    println("residual: ",Network(U1*S1*V1',U2*S2*V2',x,y))
    counter +=1;
end

fig, ax = subplots()
ax[:plot](collect(1:length(sdHistory)),sdHistory, "k-", linewidth=2, label="sd", alpha=0.6)
ax[:plot](collect(1:length(DLRHistory)),DLRHistory, "r--", linewidth=2, label="DLR", alpha=0.6)
ax[:legend](loc="upper right")
ax.set_yscale("log")
ax.tick_params("both",labelsize=20) 
show()