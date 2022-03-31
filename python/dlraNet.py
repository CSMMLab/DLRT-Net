import torch
import torch.nn.functional as F
from torch import nn

import itertools

print(torch.__version__)
# define the network
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class DLRANet(nn.Module):
    weights_K: list  # [torch.Tensor]  # n1 x r
    weights_Lt: list  # [torch.Tensor]  # n2 x r
    weights_S: list  # [torch.Tensor]  # r x r
    aux_U: list  # [torch.Tensor]  # n1 x r
    aux_Unp1: list  # [torch.Tensor]  # n1 x r
    aux_Vt: list  # [torch.Tensor]  # r x n2
    aux_Vtnp1: list  # [torch.Tensor]  # r x n2
    aux_N: list  # [torch.Tensor]  # r x r
    aux_M: list  # [torch.Tensor]  # r x r
    biases: list  # [torch.Tensor]  # n2
    num_layers: int
    layer_width: int

    optim_K: torch.optim.SGD  # optimizer for K step
    optim_Lt: torch.optim.SGD  # optimizer for L step
    optim_S: torch.optim.SGD  # optimizer for S step

    def __init__(self, input_dim: int, output_dim: int, layer_width: int, num_layers: int, low_rank: int = 10):
        self.num_layers = num_layers
        self.layer_width = layer_width
        # weight initialization
        W = torch.rand(input_dim, layer_width)
        W = W / torch.norm(W)
        # print(W.size())
        u, s, v = torch.svd(W)
        # print(W)
        # print(torch.mm(u, torch.mm(torch.diag(s), torch.transpose(v, 0, 1))))
        # print(torch.mm(torch.narrow(u, 1, 0, low_rank),
        #               torch.mm(torch.narrow(torch.narrow(torch.diag(s), 0, 0, low_rank), 1, 0, low_rank),
        #                        torch.narrow(torch.transpose(v, 0, 1), 0, 0, low_rank))))
        # print(torch.mm(torch.narrow(u, 1, 0, low_rank),
        #               torch.mm(torch.narrow(torch.narrow(torch.diag(s), 0, 0, low_rank), 1, 0, low_rank),
        #                        torch.narrow(torch.transpose(v, 0, 1), 0, 0, low_rank))) - W)
        # print(u.size())
        # print(s.size())
        # print(v.size())
        # t = torch.narrow(torch.narrow(torch.diag(s), 0, 0, low_rank) , 1, 0, low_rank)
        # print(t)
        vt = torch.transpose(v, 0, 1)
        self.weights_K: list[torch.Tensor] = [torch.rand(input_dim, low_rank)]  # gets overwritten in K-step
        self.weights_Lt: list[torch.Tensor] = [torch.rand(layer_width, low_rank)]  # gets overwritten in L-step
        self.weights_S: list[torch.Tensor] = [
            torch.narrow(torch.narrow(torch.diag(s), 0, 0, low_rank), 1, 0, low_rank)]  # narrow to rxr
        self.aux_U: list[torch.Tensor] = [torch.narrow(u, 1, 0, low_rank)]  # narrow to n2xr
        self.aux_Unp1: list[torch.Tensor] = [torch.rand(input_dim, low_rank)]  # gets overwritten
        self.aux_Vt: list[torch.Tensor] = [torch.narrow(vt, 0, 0, low_rank)]  # narrow to rxn1
        self.aux_Vtnp1: list[torch.Tensor] = [torch.rand(low_rank, layer_width)]  # gets overwritten
        self.aux_N: list[torch.Tensor] = [torch.rand(low_rank, low_rank)]  # gets overwritten
        self.aux_M: list[torch.Tensor] = [torch.rand(low_rank, low_rank)]  # gets overwritten
        self.biases: list[torch.Tensor] = [torch.rand(layer_width)]
        for i in range(1, num_layers - 1):
            # weight initialization
            self.W = torch.rand(layer_width, layer_width)
            self.W = self.W / torch.norm(self.W)
            u, s, v = torch.svd(self.W)
            vt = torch.transpose(v, 0, 1)
            self.weights_K.append(torch.rand(layer_width, low_rank, requires_grad=True))
            self.weights_Lt.append(torch.rand(layer_width, low_rank, requires_grad=True))
            self.weights_S.append(torch.narrow(torch.narrow(torch.diag(s), 0, 0, low_rank), 1, 0, low_rank))
            self.aux_U.append(torch.narrow(u, 1, 0, low_rank))
            self.aux_Unp1.append(torch.rand(layer_width, low_rank, requires_grad=False))
            self.aux_Vt.append(torch.narrow(vt, 0, 0, low_rank))
            self.aux_Vtnp1.append(torch.rand(low_rank, layer_width, requires_grad=False))
            self.aux_N.append(torch.rand(low_rank, low_rank, requires_grad=False))
            self.aux_M.append(torch.rand(low_rank, low_rank, requires_grad=False))
            self.biases.append(torch.rand(layer_width, requires_grad=True))
        W = torch.rand(layer_width, output_dim)
        W = W / torch.norm(W)
        u, s, v = torch.svd(W)
        vt = torch.transpose(v, 0, 1)
        self.weights_K.append(torch.rand(layer_width, low_rank, requires_grad=True))
        self.weights_Lt.append(torch.rand(output_dim, low_rank, requires_grad=True))
        self.weights_S.append(torch.narrow(torch.narrow(torch.diag(s), 0, 0, low_rank), 1, 0, low_rank))
        self.aux_U.append(torch.narrow(u, 1, 0, low_rank))
        self.aux_Unp1.append(torch.rand(layer_width, low_rank, requires_grad=False))
        self.aux_Vt.append(torch.narrow(vt, 0, 0, low_rank))
        self.aux_Vtnp1.append(torch.rand(low_rank, output_dim, requires_grad=False))
        self.aux_N.append(torch.rand(low_rank, low_rank, requires_grad=False))
        self.aux_M.append(torch.rand(low_rank, low_rank, requires_grad=False))
        self.biases.append(torch.rand(output_dim, requires_grad=True))

        with torch.no_grad():
            for i in range(0, self.num_layers):
                # mark for auto differentiation tape
                self.weights_K[i].requires_grad = True
                self.weights_Lt[i].requires_grad = True
                self.weights_S[i].requires_grad = True
                self.aux_U[i].requires_grad = False
                self.aux_Vt[i].requires_grad = False
                self.aux_N[i].requires_grad = False
                self.aux_M[i].requires_grad = False
                self.biases[i].requires_grad = True

        # Create optimizers
        self.optim_K = torch.optim.SGD(self.weights_K, lr=1e-3)
        self.optim_Lt = torch.optim.SGD(self.weights_Lt, lr=1e-3)
        self.optim_S = torch.optim.SGD(self.weights_S, lr=1e-3)

    def K_step_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # K-step of DRLA (forward pass)
        # prepare  K
        with torch.no_grad():
            for i in range(0, self.num_layers):
                # a) update K
                self.weights_K[i] = torch.matmul(self.aux_U[i], self.weights_S[i])
                self.weights_K[i].requires_grad = True
        z = input_tensor
        # pass forward
        for i in range(0, self.num_layers - 1):
            # z = f(xW+b) = f(xKV + b)
            z = F.relu(torch.matmul(z, torch.matmul(self.weights_K[i], self.aux_Vt[i])))

        return F.log_softmax(
            torch.matmul(z, torch.matmul(self.weights_K[self.num_layers - 1], self.aux_Vt[self.num_layers - 1])), -1)

    def K_step_update(self):
        # K-step of DRLA (update)

        # 1) Apply optimizer t K matrix
        self.optim_K.step()

        # 2) Update auxiliary matrices Unp1 and N
        with torch.no_grad():
            for i in range(0, self.num_layers):
                # gradient update
                # self.weights_K[i] = self.weights_K[i] - stepsize * self.weights_K[i].grad
                self.weights_K[i].requires_grad = True
                # Create U
                self.aux_Unp1[i], _ = torch.qr(self.weights_K[i])
                # print(self.weights_K[i].size())
                # print(t.size())
                # print("unp1")
                # print(self.aux_Unp1[i].size())
                # print("U")
                # print(self.aux_U[i].size())
                # print("prod")
                # tmp = torch.matmul(torch.transpose(self.aux_Unp1[i], 0, 1),self.aux_U[i])
                # print(tmp.size())
                # print("____")
                # Create N
                self.aux_N[i] = torch.matmul(torch.transpose(self.aux_Unp1[i], 0, 1), self.aux_U[i])
        return None

    def L_step_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # L-step of DLRA (forward)
        # prepare  L
        with torch.no_grad():
            for i in range(0, self.num_layers):
                # a) update L
                self.weights_Lt[i] = torch.matmul(self.weights_S[i], self.aux_Vt[i])  # L-transpose
                self.weights_Lt[i].requires_grad = True
        z = input_tensor
        # pass forward
        for i in range(0, self.num_layers - 1):
            # z = f(xW+b) = f(xUL + b)
            z = F.relu(torch.matmul(z, torch.matmul(self.aux_U[i], self.weights_Lt[i])))
        return F.log_softmax(
            torch.matmul(z, torch.matmul(self.aux_U[self.num_layers - 1], self.weights_Lt[self.num_layers - 1])), -1)

    def L_step_update(self):
        # L-step of DRLA (update)
        # 1) Apply optimizer to Lt matrix
        self.optim_Lt.step()

        # 2) Update auxiliary matrices Vtnp1 and M
        with torch.no_grad():
            for i in range(0, self.num_layers):
                # gradient update
                # self.weights_Lt[i] = self.weights_Lt[i] - stepsize * self.weights_Lt[i].grad
                self.weights_Lt[i].requires_grad = True
                # Create V_np1
                self.aux_Vtnp1[i], _ = torch.qr(torch.transpose(self.weights_Lt[i], 0, 1))
                self.aux_Vtnp1[i] = torch.transpose(self.aux_Vtnp1[i], 0, 1)
                # Create N
                # print("S")
                # print(self.weights_S[i].size())
                # print("V")
                # print(self.aux_Vt[i].size())
                # print("L")
                # print(self.weighweights_Ltts_L[i].size())
                # print("t")
                # print(t.size())
                # print("Vnp1")
                # print(self.aux_Vtnp1[i].size())
                # print("V")
                # print(self.aux_Vt[i].size())
                # print("prod")
                # tmp = torch.matmul(self.aux_Vtnp1[i], torch.transpose(self.aux_Vt[i], 0, 1))
                # print(tmp.size())
                # print("____")
                self.aux_M[i] = torch.matmul(self.aux_Vtnp1[i], torch.transpose(self.aux_Vt[i], 0, 1))  # Vtnp1*V
                # Update U_np1
                # self.aux_U[i] = U_np1
        return None

    def S_step_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # S-step of DLRA (forward)
        # prepare  S
        with torch.no_grad():
            for i in range(0, self.num_layers):
                # a) update S
                self.weights_S[i] = torch.matmul(torch.matmul(self.aux_N[i], self.weights_S[i]),
                                                 torch.transpose(self.aux_M[i], 0, 1))  # S
                self.weights_S[i].requires_grad = True
                # if i == self.num_layers - 2:
                # S_tilde = torch.matmul(torch.matmul(self.aux_N[i], self.weights_S[i]), torch.transpose(self.aux_M[i], 0, 1))  # S
                # W = torch.matmul(self.aux_Unp1[i], torch.matmul(S_tilde, self.aux_Vtnp1[i]))
                # print("U-N-Ut identity check")
                # u_id = torch.matmul(torch.matmul(self.aux_Unp1[i], self.aux_N[i]),
                #                    torch.transpose(self.aux_U[i], 0, 1))
                # print(u_id - torch.eye(self.layer_width))
                # print("Vt-M-V identity check")
                # v_id = torch.matmul(
                #    torch.matmul(torch.transpose(self.aux_Vt[i], 0, 1), torch.transpose(self.aux_M[i], 0, 1)),
                #    self.aux_Vtnp1[i])
                # print(v_id - torch.eye(self.layer_width))
                # print("W= Unp1NSMtVtnp1 identity check")
                # print(self.W - W)
                # else:

        z = input_tensor
        # pass forward
        for i in range(0, self.num_layers - 1):
            # z = f(xW+b) \approx f(xUnp1 S Vnp1^T + b)
            z = F.relu(
                torch.matmul(z, torch.matmul(self.aux_Unp1[i], torch.matmul(self.weights_S[i], self.aux_Vtnp1[i]))))
        # print(self.W.size())
        # print(self.W)
        # t = torch.matmul(self.aux_Unp1[self.num_layers - 2],
        #                 torch.matmul(self.weights_S[self.num_layers - 2], self.aux_Vtnp1[self.num_layers - 2]))
        # print(t.size())
        # print(t)
        # print("____")
        return F.log_softmax(torch.matmul(z, torch.matmul(self.aux_Unp1[self.num_layers - 1],
                                                          torch.matmul(self.weights_S[self.num_layers - 1],
                                                                       self.aux_Vtnp1[self.num_layers - 1]))), -1)

    def S_step_update(self, ):
        # S-step of DRLA (update)
        # 1) Apply optimizer to S matrix
        self.optim_S.step()

        # 2) Update auxiliary matrices Vtnp1 and M
        with torch.no_grad():
            for i in range(0, self.num_layers):
                # gradient update
                # self.weights_S[i] = self.weights_S[i] - stepsize * self.weights_S[i].grad
                self.weights_S[i].requires_grad = True
                # update U to Unp1 and V to Vnp1
                self.aux_U[i] = self.aux_Unp1[i]
                self.aux_Vt[i] = self.aux_Vtnp1[i]
        return None

    def clear_grads(self):
        # print("Clear Grads")
        for i in range(0, self.num_layers):
            if self.weights_K[i].grad is not None:
                self.weights_K[i].grad.data.zero_()
            if self.weights_Lt[i].grad is not None:
                self.weights_Lt[i].grad.data.zero_()
            if self.weights_S[i].grad is not None:
                self.weights_S[i].grad.data.zero_()
            if self.aux_U[i].grad is not None:
                self.aux_U[i].grad.data.zero_()
            if self.aux_Unp1[i].grad is not None:
                self.aux_Unp1[i].grad.data.zero_()
            if self.aux_Vt[i].grad is not None:
                self.aux_Vt[i].grad.data.zero_()
            if self.aux_Vtnp1[i].grad is not None:
                self.aux_Vtnp1[i].grad.data.zero_()
            if self.aux_N[i].grad is not None:
                self.aux_N[i].grad.data.zero_()
            if self.aux_M[i].grad is not None:
                self.aux_M[i].grad.data.zero_()
            if self.biases[i].grad is not None:
                self.biases[i].grad.data.zero_()
        return None

    def print_layer_size(self):
        for i in range(0, self.num_layers):
            print("layer " + str(i))
            print("K")
            print(self.weights_K[i].size())
            print("Lt")
            print(self.weights_Lt[i].size())
            print("S")
            print(self.weights_S[i].size())
            print("U")
            print(self.aux_U[i].size())
            print("Unp1")
            print(self.aux_Unp1[i].size())
            print("Vt")
            print(self.aux_Vt[i].size())
            print("Vtnp1")
            print(self.aux_Vtnp1[i].size())
            print("N")
            print(self.aux_N[i].size())
            print("M")
            print(self.aux_M[i].size())
            print("b")
            print(self.biases[i].size())

    def print_layer_weights(self):
        for i in range(0, self.num_layers):
            print("layer " + str(i))
            print("K")
            print(self.weights_K[i])
            print("L")
            print(self.weights_Lt[i])
            print("S")
            print(self.weights_S[i])
            print("U")
            print(self.aux_U[i])
            print("Unp1")
            print(self.aux_Unp1[i])
            print("Vt")
            print(self.aux_Vt[i])
            print("Vtnp1")
            print(self.aux_Vtnp1[i])
            print("N")
            print(self.aux_N[i])
            print("M")
            print(self.aux_M[i])
            print("b")
            print(self.biases[i])
        return None

    def print_weights_K(self):
        print("K")
        for i in range(0, self.num_layers):
            print("layer " + str(i))
            print(self.weights_K[i])

    def print_weights_Lt(self):
        print("L")
        for i in range(0, self.num_layers):
            print("layer " + str(i))
            print(self.weights_Lt[i])

    def print_weights_S(self):
        print("S")
        for i in range(0, self.num_layers):
            print("layer " + str(i))
            print(self.weights_S[i])

    def print_aux_M(self):
        print("M")
        for i in range(0, self.num_layers):
            print("layer " + str(i))
            print(self.aux_M[i])

    def print_aux_N(self):
        print("N")
        for i in range(0, self.num_layers):
            print("layer " + str(i))
            print(self.aux_N[i])


if __name__ == '__main__':
    ## some sanity checks
    test_net = DLRANet(input_dim=10, output_dim=10, layer_width=10, num_layers=3, low_rank=10)
    # test_net.print_layer_size()
    print("input")
    x = torch.rand(5, 10)  # random flattened images
    y = torch.randint(0, 9, (5,))  # random labels
    # print(x)
    # print(x.size())
    # print("labels")
    # print(y.size())

    print("K_step")
    out = test_net.K_step_forward(x)
    # print("output")
    # print(out.size())
    loss = F.nll_loss(out, y)
    # print("loss")
    # print(loss.size())
    # print(loss)
    loss.backward()
    test_net.K_step_update(stepsize=1e-2)
    test_net.clear_grads()

    print("L_step_forward")
    out = test_net.L_step_forward(x)
    # print("output")
    # print(out.size())
    loss = F.nll_loss(out, y)
    # print("loss")
    # print(loss.size())
    # print(loss)
    loss.backward()
    test_net.L_step_update(stepsize=1e-2)
    test_net.clear_grads()

    print("S_step_forward")
    out = test_net.S_step_forward(x)
    print("output")
    print(out.size())
    loss = F.nll_loss(out, y)
    print("loss")
    print(loss.size())
    print(loss)
    loss.backward()
    test_net.S_step_update(stepsize=1e-2)
    test_net.clear_grads()
