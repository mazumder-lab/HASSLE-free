import math
import time
import copy 
import gc
import torch
import torch.nn as nn
from peft.tuners import lora
import torch.nn.functional as F
from peft.tuners import lora
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def randomized_svd(
    A: torch.Tensor,
    num_ranks: int = 64,
    num_oversampling: int = 5,
):
    if A.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim}.")

    U, S, V = torch.svd_lowrank(A, num_ranks + num_oversampling)
    # https://pytorch.org/docs/stable/_modules/torch/_lowrank.html#svd_lowrank
    VT = V.mH

    S_sqrt = torch.sqrt(S)
    L1 = U * S_sqrt.unsqueeze(dim=0)
    L2 = VT * S_sqrt.unsqueeze(dim=1)
    L1k = L1[:, :num_ranks]
    L2k = L2[:num_ranks, :]
    return L2k, L1k


def low_rank_adam(rescaled_XTX, rescaled_W_diff, init_A, init_B, max_iter=200, lr_init=1e-4):
    # Minimize ||XW_old - X(W_2_4 + AB)||_F^2
    with torch.enable_grad():
        A = init_A.clone()
        B = init_B.clone()
        A.requires_grad = True
        B.requires_grad = True
        # Initialize the Adam optimizer with A and B as parameters
        optimizer = torch.optim.Adam([A, B], lr=lr_init)
        for it in range(max_iter):
            optimizer.zero_grad()
            # Compute W_diff and the loss
            W_diff = rescaled_W_diff - B @ A
            loss = 1/2 * torch.trace(W_diff @ rescaled_XTX @ W_diff.t())
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Iteration {it + 1}, encountered NaN or Inf in loss. Restarting with reduced lr.")
                # Reset parameters and reduce learning rate
                lr_init /= 10
                A = init_A.clone()
                B = init_B.clone()
                optimizer = torch.optim.Adam([A, B], lr=lr_init)
                continue
            # Print the loss every 100 iterations or on the first iteration
            if (it + 1) % 50 == 0 or it == 0:
                print(f"Iteration {it + 1}, Real Loss: {loss.item()}")
            # Backpropagation
            loss.backward()
            optimizer.step()
        return A.detach(), B.detach()


def sparsegpt_prune(H, Hinv, W_diff, layer, device, prunen=2, prunem=4, blocksize=128, sparsity=None):
    print("Starting SparseGPT-Prune. Displaying pruning parameters")
    print("prunen", prunen)
    print("prunem", prunem)
    print("sparsity", sparsity)
    print("W_diff shape", W_diff.shape)
    prunen = prunem - prunen
    
    W = W_diff.clone().float()
    rows = W.shape[0]
    columns = W.shape[1]

    tick = time.time()
    
    dead = torch.diag(H) == 0
    W[:, dead] = 0
    Losses = torch.zeros(rows, device=device)

    mask = None

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if prunen == 0: 
            if mask is not None:
                mask1 = mask[:, i1:i2]
            else:
                tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
        else:
            mask1 = torch.zeros_like(W1) == 1

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if prunen != 0 and i % prunem == 0:
                tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

            q = w.clone()
            q[mask1[:, i]] = 0

            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        Losses += torch.sum(Losses1, 1) / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
    torch.cuda.synchronize()
    print('time %.2f' % (time.time() - tick))
    print('error', torch.sum(Losses).item())
        
    return W.reshape(layer.weight.shape).to(layer.weight.data.dtype)


def cg_batch(A, B, A_supp, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

    This function solves matrix linear systems of the form

        A X = B,  

    where A is a n x n positive definite matrix and B is a n x m matrix,
    and X is the n x m matrix representing the solution for the ith system.

    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a n x m matrix.
        B: A n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
        matrices M and a n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    error_list = np.zeros((maxiter,))
    n, m = B.shape
    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n
    assert B.shape == (n, m)
    assert X0.shape == (n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)
    X_k = X0
    R_k = B - A @ X_k
    R_k = R_k * A_supp
    Z_k = M_bmm(R_k)
    P_k = torch.zeros_like(Z_k)
    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k
    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))
    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))
    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)
        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(0)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(0) / denominator
            P_k = Z_k1 + beta.unsqueeze(0) * P_k1
        denominator = (P_k * (A@P_k)).sum(0)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(0) / denominator
        X_k = X_k1 + alpha.unsqueeze(0) * P_k
        R_k = R_k1 - alpha.unsqueeze(0) * (A@P_k)
        R_k = R_k * A_supp
        end_iter = time.perf_counter()
        residual_norm = torch.norm(A@X_k - B, dim=1)
        if verbose:
            print("%03d | %8.4e" %
                    (k, torch.max(residual_norm/B_norm)))
        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break
    end = time.perf_counter()
    if verbose:
        if optimal:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                    (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                    (k, (end - start) * 1000))
    info = {
        "niter": k,
        "optimal": optimal
    }
    return X_k


def alps_prune(XtX, X_norm, L, Q,  W_diff, layer, dev, nm_n = 2, nm_m = 4, sp = 0.0, rho=0.1, max_iter = 200, update_iter = 3, switch_iter = 30):
    W = W_diff.clone().float().to(dev)
    YtX = torch.zeros_like(W)
    YtX = torch.matmul(W * X_norm, XtX).to(dev)
    admm_st = time.time()
    XTX_inv = torch.zeros_like(XtX).float().to(dev)
    B = (W * X_norm.to(dev)).t().clone()
    W = None
    B_orig = B.clone()
    V = torch.zeros_like(B)
    D = torch.zeros_like(B)
    D_suppp = torch.zeros_like(B)
    D_supp = torch.zeros_like(B)
    totp, num_cout = B.shape
    XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(dev)
    init_rho = False
    fix_supp = False
    D_fix = torch.zeros_like(D)
    Res0 = YtX.T
    Res0 = torch.sum(B_orig * Res0)
    Res0 = torch.sum(Res0)
    params = B.shape[0]*B.shape[1]
    if nm_n == 0:
        k_spar = int(np.round((1-sp)*params))
        D = B.clone().reshape(-1)
        _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
        D[loss_idx] = 0    
        D_suppp = (D == 0).to(torch.float)
        D = D.reshape(totp, num_cout)
    else:
        new_dim = totp * num_cout / nm_m
        new_dim = int(new_dim)
        k_spar = totp * num_cout * nm_n/nm_m
        D = B.clone().t().reshape((new_dim, nm_m))
        _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
        D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to('cuda:0'),dim=1,index=loss_idx)   
        D_suppp = (D == 0).to(torch.float)
        D = D.reshape(num_cout, totp).t()
    D_init = D.clone()
    errorp = 1
    for i_admm in range(max_iter):
        B = XTX_inv @ (YtX.T-V+rho*D)
        if fix_supp:
            D = ((V + rho * B) / rho) * D_fix
        elif nm_n == 0:
            D = ((V + rho * B) / rho).reshape(-1)
            _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
            D[loss_idx] = 0    
            D = D.reshape(totp, num_cout)   
        else:
            D = ((V + rho * B) / rho).t().reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
            D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to('cuda:0'),dim=1,index=loss_idx) 
            D_supp = (D == 0).to(torch.float)  
            D = D.reshape(num_cout, totp).t()  
        V = V + rho * (B - D)
        if (i_admm+1) % update_iter == 0:
            if nm_n == 0:
                D_supp = (D.reshape(-1) == 0).to(torch.float)
            supp_change = torch.sum((D_supp-D_suppp)**2)
            
            if not fix_supp:
                if supp_change / k_spar > 0.1:
                    init_rho = True
                    rho *= 1.3
                elif supp_change / k_spar > 0.005:
                    init_rho = True
                    rho *= 1.2
                elif supp_change > 0.5:
                    if init_rho:
                        rho *= 1.1
                    else:
                        rho /= 5
                        B = B_orig.clone().to(dev)
                        D = D_init.clone().to(dev)
                        V = torch.zeros_like(B).to(dev)     
                else:
                    if init_rho:
                        break
                    else:
                        rho /= 5
            D_suppp = (D_supp).clone()
            if rho > 1e6:
                rho = 1e6
            XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(dev)
            if nm_n == 0:
                Btest = B.reshape(-1)
                _, loss_idx = torch.topk(-Btest**2,totp * num_cout - k_spar)
                Btest[loss_idx] = 0    
                Btest = Btest.reshape(totp, num_cout)
            else:
                Btest = B.t().reshape((new_dim, nm_m))
                _, loss_idx = torch.topk(-Btest**2,nm_m - nm_n, dim = 1)
                Btest = Btest.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to('cuda:0'),dim=1,index=loss_idx)  
                Btest = Btest.reshape(num_cout, totp).t()
            Resc = torch.matmul(XtX.to(dev),Btest) - YtX.T
            Resc = torch.diag(torch.matmul((Btest-B_orig.to(dev)).t(), Resc))
            errorc = torch.sum(Resc).to("cpu")/Res0
            errorc = errorc.item()
            print("iter {}, error {} support change {}, rho {}".format(i_admm, errorc / errorp, supp_change / k_spar, rho))
            if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                break
    if nm_n == 0:
        B = B.reshape(-1)
        _, loss_idx = torch.topk(-B**2,totp * num_cout - k_spar)
        B[loss_idx] = 0    
        B = B.reshape(totp, num_cout)
    else:
        B = B.t().reshape((new_dim, nm_m))
        _, loss_idx = torch.topk(-B**2,nm_m - nm_n, dim = 1)
        B = B.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to('cuda:0'),dim=1,index=loss_idx)  
        B = B.reshape(num_cout, totp).t()
    V = None
    D = None
    Res = torch.matmul(XtX,B ) - YtX.T
    Res = torch.diag(torch.matmul((B  -B_orig).t(), Res))
    error = torch.sum(Res)/Res0
    error = error.item()
    print("Before backsolve, error is {}".format(error))
    admm_time = time.time() - admm_st
    back_st = time.time()
    B = cg_batch((XtX).to(dev), YtX.T, 
                    (B != 0).to(torch.float), M_bmm=None, X0=B, rtol=1e-4, atol=0., maxiter=10, verbose=True)
    back_time = time.time() - back_st
    Res = torch.matmul(XtX,B ) - YtX.T
    Res = torch.diag(torch.matmul((B  -B_orig).t(), Res))
    error = torch.sum(Res)/Res0
    error = error.item()
    torch.cuda.synchronize()
    print("Number of iter is {}".format(i_admm))
    print("Final Error is {}".format(error))
    print("Time is admm: {} back:{}".format(admm_time, back_time))    
    return (B.t() / X_norm.to(dev)).reshape(layer.weight.shape).to(layer.weight.data.dtype)


def gptq_quantize(H, Hinv, W_diff, layer, device, blocksize=128, groupsize=-1, actorder=False, static_groups=False):
    print("Starting GPTQ-Quantize. Displaying pruning parameters")
    print("W_diff shape", W_diff.shape)
    
    W = W_diff.clone().float()
    rows = W.shape[0]
    columns = W.shape[1]

    tick = time.time()
    
    dead = torch.diag(H) == 0
    W[:, dead] = 0
    Losses = torch.zeros(rows, device=device)

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    if static_groups:
        import copy
        groups = []
        for i in range(0, columns, groupsize):
            quantizer = copy.deepcopy(quantizer)
            quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
            groups.append(quantizer)

    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if groupsize != -1:
                if not static_groups:
                    if (i1 + i) % groupsize == 0:
                        quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                else:
                    idx = i1 + i
                    if actorder:
                        idx = perm[idx]
                    quantizer = groups[idx // groupsize]

            q = quantize(
                w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
            ).flatten()
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    torch.cuda.synchronize()
    print('time %.2f' % (time.time() - tick))
    print('error', torch.sum(Losses).item())

    if actorder:
        Q = Q[:, invperm]
        
    torch.cuda.synchronize()
    print('time %.2f' % (time.time() - tick))
    print('error', torch.sum(Losses).item())
        
    return W.reshape(layer.weight.shape).to(layer.weight.data.dtype)



class LLM_AM_Compressor:
    """Compressing a Model in 2:4_sparse + LR through Alternating Minimization. 
    Class takes as input a layer from lora_model. It has weight A and B components randomly initialized.
    Three major steps are to be done.
        - Calculate XTX in add_batch
        - Alternate Minimization between
            - 2:4_sparse with SparseGPT/ALPS/WANDA
            - Low-Rank with simple R-SVD/Gradient Descent with ||X(W - W')||_F^2"""
    def __init__(self, layer, name):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        W = None
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        
        self.scaler_row = torch.zeros((self.columns), device=self.dev)

    def add_batch(self, inp, out, blocksize=1024):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, lora.LoraLayer):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1])) #4096x4096
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.to(dtype=self.H.dtype)
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        
        self.scaler_row += torch.norm(inp.clone(), p=2, dim=1) ** 2  / self.nsamples

    @torch.no_grad()
    def scale_sparsegpt_gd(self, n_iters=5, prunen=0, prunem=0, sparsity=0.0, percdamp=0.01, max_iter=50, lr_init=1e-2, hess_diag=False, hess_percdamp=0.05):
        start_hess_inv = time.time()
        module = self.layer
        W_old = module.base_layer.weight.data.clone()
        scale_sqrt = math.sqrt(module.scaling[module.active_adapter[0]])
        A = module.lora_A.default.weight.clone() * scale_sqrt
        B = module.lora_B.default.weight.clone() * scale_sqrt     
        print("Starting SparseGPT-GD")
        print("W_old shape", W_old.shape)
        print("A shape", A.shape)
        print("B shape", B.shape)
        H = self.H
        dead = torch.diag(H) == 0
        print("dead:", torch.where(dead == True))
        H[dead, dead] = 1
        rows = W_old.shape[0]
        columns = W_old.shape[1]
        if not hess_diag:
            damp = percdamp * torch.mean(torch.diag(H))
        else:
            damp = hess_percdamp * torch.diag(H) + percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        XtX = H.clone()
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        print(f"Time to calculate Hinv: {time.time() - start_hess_inv}")
        best_W_2_4 = None
        best_A = None
        best_B = None
        best_loss = float('inf')
        
        X_norm = torch.diag(XtX).sqrt()
        scaled_XtX = XtX / X_norm
        scaled_XtX = (scaled_XtX.T / X_norm).T
        for it in range(n_iters):
            if it == 10:
                start_inner_time = time.time()
            W_2_4 = sparsegpt_prune(self.H, Hinv, W_old - B @ A, self.layer, self.dev, prunen=prunen, prunem=prunem, sparsity=sparsity)
            if it == 10:
                mid_inner_time = time.time()
                print(f"Sparse update time: {mid_inner_time - start_inner_time}")
            A, B = low_rank_adam(scaled_XtX, (W_old - W_2_4) * X_norm, A * X_norm, B, lr_init=lr_init/(it+10), max_iter=max_iter)
            if it == 10:
                end_inner_time = time.time()
                print(f"Low-Rank update time: {end_inner_time - mid_inner_time}")
            A = A / X_norm
            M_diff = W_old - W_2_4 - (B @ A)
            diff_norm = torch.norm(M_diff, p="fro")
            true_loss = 1/2 * torch.trace(M_diff @ self.H @ M_diff.t())
            print(f"For iteration {it}, the data-free distance ||W_old - (W_2_4 + B @ A)||_F^2 = {diff_norm}")
            print(f"For iteration {it}, the true loss ||X(W_old - (W_2_4 + B @ A))||_F^2 = {true_loss}")
            if true_loss < best_loss:
                best_loss = true_loss
                best_W_2_4 = W_2_4.clone()
                best_A = A.clone()
                best_B = B.clone()
        module.base_layer.weight.copy_(best_W_2_4)
        module.lora_A.default.weight.copy_(best_A / scale_sqrt)
        module.lora_B.default.weight.copy_(best_B / scale_sqrt)
        
    @torch.no_grad()
    def scale_alps_gd(self, n_iters=5, prunen=0, prunem=0, sparsity=0.0, percdamp=0.01, max_iter=50, lr_init=1e-2, hess_diag=False, hess_percdamp=0.05):
        start_hess_inv = time.time()
        module = self.layer
        W_old = module.base_layer.weight.data.clone()
        scale_sqrt = math.sqrt(module.scaling[module.active_adapter[0]])
        A = module.lora_A.default.weight.clone() * scale_sqrt
        B = module.lora_B.default.weight.clone() * scale_sqrt     
        print("Starting SparseGPT-GD")
        print("W_old shape", W_old.shape)
        print("A shape", A.shape)
        print("B shape", B.shape)
        H = self.H
        dead = torch.diag(H) == 0
        print("dead:", torch.where(dead == True))
        H[dead, dead] = 1
        rows = W_old.shape[0]
        columns = W_old.shape[1]
        if not hess_diag:
            damp = percdamp * torch.mean(torch.diag(H))
        else:
            damp = hess_percdamp * torch.diag(H) + percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        XtX = H.clone()
        print(f"Time to calculate Hinv: {time.time() - start_hess_inv}")
        best_W_2_4 = None
        best_A = None
        best_B = None
        best_loss = float('inf')
        
        X_norm = torch.diag(XtX).sqrt() + 1e-9
        scaled_XtX = XtX / X_norm
        scaled_XtX = (scaled_XtX.T / X_norm).T
        L, Q = torch.linalg.eigh(scaled_XtX.double())

        for it in range(n_iters):
            if it == 10:
                start_inner_time = time.time()
            W_2_4 = alps_prune(scaled_XtX, X_norm, L, Q,  W_old - B @ A, self.layer, self.dev, nm_n=prunen, nm_m=prunem, sp=sparsity, rho=0.1*(5*it+1))            
            if it == 10:
                mid_inner_time = time.time()
                print(f"Sparse update time at iter10: {mid_inner_time - start_inner_time}")
            A, B = low_rank_adam(scaled_XtX, (W_old - W_2_4) * X_norm, A * X_norm, B, lr_init=lr_init/(it+10), max_iter=max_iter)
            if it == 10:
                end_inner_time = time.time()
                print(f"Low-Rank update time at iter10: {end_inner_time - mid_inner_time}")
            A = A / X_norm
            M_diff = W_old - W_2_4 - (B @ A)
            diff_norm = torch.norm(M_diff, p="fro")
            true_loss = 1/2 * torch.trace(M_diff @ self.H @ M_diff.t())
            print(f"For iteration {it}, the data-free distance ||W_old - (W_2_4 + B @ A)||_F^2 = {diff_norm}")
            print(f"For iteration {it}, the true loss ||X(W_old - (W_2_4 + B @ A))||_F^2 = {true_loss}")
            if true_loss < best_loss:
                best_loss = true_loss
                best_W_2_4 = W_2_4.clone()
                best_A = A.clone()
                best_B = B.clone()
        module.base_layer.weight.copy_(best_W_2_4)
        module.lora_A.default.weight.copy_(best_A / scale_sqrt)
        module.lora_B.default.weight.copy_(best_B / scale_sqrt)

    def free(self):
        self.H = None
        gc.collect()
        torch.cuda.empty_cache()