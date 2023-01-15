import torch
import torch.nn as nn
import numpy as np
import math

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_cc_weights(nb_steps):
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = torch.tensor(lam.T @ W).float()
    steps = torch.tensor(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)).float()

    return cc_weights, steps


def integrate(x0, nb_steps, step_sizes, integrand, h, compute_grad=False, x_tot=None):
    #Clenshaw-Curtis Quadrature Method
    cc_weights, steps = compute_cc_weights(nb_steps)

    device = x0.device
    cc_weights, steps = cc_weights.to(device), steps.to(device)

    xT = x0 + nb_steps*step_sizes
    if not compute_grad:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t-x0_t)*(steps_t + 1)/2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        dzs = integrand(X_steps, h_steps)
        dzs = dzs.view(xT_t.shape[0], nb_steps+1, -1)
        dzs = dzs*cc_weights.unsqueeze(0).expand(dzs.shape)
        z_est = dzs.sum(1)
        return z_est*(xT - x0)/2
    else:

        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        x_tot = x_tot * (xT - x0) / 2
        x_tot_steps = x_tot.unsqueeze(1).expand(-1, nb_steps + 1, -1) * cc_weights.unsqueeze(0).expand(x_tot.shape[0], -1, x_tot.shape[1])
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        x_tot_steps = x_tot_steps.contiguous().view(-1, x_tot.shape[1])

        g_param, g_h = computeIntegrand(X_steps, h_steps, integrand, x_tot_steps, nb_steps+1)
        return g_param, g_h


def computeIntegrand(x, h, integrand, x_tot, nb_steps):
    h.requires_grad_(True)
    with torch.enable_grad():
        f = integrand.forward(x, h)
        g_param = _flatten(torch.autograd.grad(f, integrand.parameters(), x_tot, create_graph=True, retain_graph=True))
        g_h = _flatten(torch.autograd.grad(f, h, x_tot))

    return g_param, g_h.view(int(x.shape[0]/nb_steps), nb_steps, -1).sum(1)


class ParallelNeuralIntegral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, h, nb_steps=20):
        with torch.no_grad():
            x_tot = integrate(x0, nb_steps, (x - x0)/nb_steps, integrand, h, False)
            # Save for backward
            ctx.integrand = integrand
            ctx.nb_steps = nb_steps
            ctx.save_for_backward(x0.clone(), x.clone(), h)
        return x_tot

    @staticmethod
    def backward(ctx, grad_output):
        x0, x, h = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        integrand_grad, h_grad = integrate(x0, nb_steps, x/nb_steps, integrand, h, True, grad_output)
        x_grad = integrand(x, h)
        x0_grad = integrand(x0, h)
        # Leibniz formula
        return -x0_grad*grad_output, x_grad*grad_output, None, integrand_grad, h_grad.view(h.shape), None


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.


class MonotonicNN(nn.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.nb_steps = nb_steps

    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h is just other conditionning variables.
    '''
    def forward(self, x, h):
        x0 = torch.zeros(x.shape, device=x.device)
        out = self.net(h)
        offset = out[:, [0]]
        scaling = torch.exp(out[:, [1]])
        return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset


class MonoRegularLayer(nn.Module):
    def __init__(self, output_dim, hidden_dims=[16, 16, 16], nb_steps=50):
        super().__init__()
        self.num_classes = output_dim
        self.umnns = nn.ModuleList()
        for i in range(self.num_classes):
            umnn = MonotonicNN(self.num_classes, hidden_dims, nb_steps=nb_steps)
            self.umnns += [umnn]

    def forward(self, x):
        """
        Args:
            node_feat_wt:   (N, L, F).
            node_feat_mut:  (N, L, F).
        """
        outs = []
        for i, umnn in enumerate(self.umnns):
            indicator = torch.zeros_like(x, device=x.device)
            indicator[..., 0] = 1
            indicator[..., i] = 1
            xx = indicator * x
            out = umnn(xx[..., [0]], xx[..., 1:])
            outs += [out]
        out = torch.cat(outs, dim=-1)
        return out