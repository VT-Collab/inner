import torch
def elbo_scores(policy, states, actions, timesteps, noise, create_graph=True):
    """Return g[B,L,n], averaging M score draws before the outer product.

    actions: [B,L,H,4], timesteps: [B,L,M], noise: [B,L,M,H,4].
    """
    b, l, m = timesteps.shape
    s = states.detach()[:, None, None].expand(b, l, m, -1).reshape(
        b * l * m, -1).clone().requires_grad_(True)
    a = actions.detach()[:, :, None].expand(b, l, m, policy.pred_horizon, 4)
    a = a.reshape(-1, policy.pred_horizon, 4)
    e = noise.detach().reshape_as(a)
    t = timesteps.reshape(-1)
    prediction = policy(policy.corrupt(a, t, e), t, s)
    # Equation (20) uses squared Euclidean norm, not coordinate-averaged MSE.
    weighted_loss = (policy.elbo_weights[t] / policy.score_t_probs[t]) * (prediction - e).square().sum((1, 2))
    gradients = torch.autograd.grad(weighted_loss.sum(), s,
                                    create_graph=create_graph)[0]
    scores = -gradients.reshape(b, l, m, -1).mean(2)
    if hasattr(policy, 'scramble'):
        scores = policy.scramble.jacobian(scores, states)
    return scores


def fisher_from_scores(scores):
    """Average over action samples, after inner score averaging (Eq. 17)."""
    return scores.transpose(-1, -2) @ scores / scores.shape[1]


def neighborhood_penalty(fisher, rank=8, k=8, power=1.0, eps=1e-8):
    """Leading-r projector KNN followed by a detached-trace tail ratio."""
    n = fisher.shape[-1]
    if not 0 < rank < n or k < 1 or not 0 < power <= 1 or eps <= 0:
        raise ValueError("Invalid spectral regularizer settings")
    with torch.no_grad():
        _, vectors = torch.linalg.eigh(fisher)
        basis = vectors[:, :, -rank:]
        projectors = basis @ basis.transpose(-1, -2)
        distances = torch.cdist(projectors.flatten(1).double(), projectors.flatten(1).double())
        nearest = distances.topk(min(k, len(fisher)), largest=False).indices
    average = fisher[nearest].mean(1)
    eigenvalues = torch.linalg.eigvalsh(average).clamp_min(0)
    bottom = eigenvalues[:, :n-rank].sum(-1)
    total = eigenvalues.sum(-1)
    return (bottom / (total.detach() + eps)).mean(), nearest


def inner_regularizer(policy, states, generator, rank=8, k=8,
                      action_samples=8, score_samples=4, power=1.0, eps=1e-8,
                      batch_average=False):
    if power != 1.0:
        raise ValueError("Backup ratio requires power=1")
    b = len(states)
    if action_samples < rank or score_samples < 1:
        raise ValueError("Need action_samples >= rank and positive score_samples")
    with torch.no_grad():
        repeated = states[:, None].expand(b, action_samples, -1).reshape(
            b * action_samples, -1)
        actions = policy.sample(repeated, generator, n_sample_steps=3).reshape(
            b, action_samples, policy.pred_horizon, 4)
    t = torch.multinomial(policy.score_t_probs, b * action_samples * score_samples,
                          replacement=True, generator=generator).reshape(b, action_samples, score_samples)
    noise = torch.randn((b, action_samples, score_samples, policy.pred_horizon, 4),
                        dtype=states.dtype, device=states.device, generator=generator)
    scores = elbo_scores(policy, states, actions, t, noise)
    fisher = fisher_from_scores(scores)
    if batch_average:
        # InneR-C: k=B, so every neighborhood is the same complete minibatch.
        # One mean/eigendecomposition avoids redundant SVD, KNN, and B copies.
        eigenvalues = torch.linalg.eigvalsh(fisher.mean(0)).clamp_min(0)
        return eigenvalues[:states.shape[-1]-rank].sum() / (eigenvalues.sum().detach() + eps)
    with torch.no_grad():
        _, _, vh = torch.linalg.svd(scores, full_matrices=False)
        basis = vh[:, :rank]
        projectors = basis.mT @ basis
        distances = torch.cdist(projectors.flatten(1).double(), projectors.flatten(1).double())
        nearest = distances.topk(min(k, b), largest=False).indices
    eigenvalues = torch.linalg.eigvalsh(fisher[nearest].mean(1)).clamp_min(0)
    penalty = (eigenvalues[:, :states.shape[-1]-rank].sum(-1) /
               (eigenvalues.sum(-1).detach() + eps)).mean()
    return penalty


def score_penalty(scores, method):
    if method=='dp_frobenius':
        return fisher_from_scores(scores).diagonal(dim1=-2,dim2=-1).sum(-1).mean()
    if method=='dp_nuclear':
        return (torch.linalg.svdvals(scores).sum(-1)/scores.shape[1]**0.5).mean()
    raise ValueError(method)


def regularizer(policy,states,generator,method,action_samples=8,score_samples=4):
    b=len(states);l=action_samples;m=score_samples
    with torch.no_grad():
        actions=policy.sample(states[:,None].expand(b,l,-1).reshape(b*l,-1),generator,n_sample_steps=3).reshape(b,l,1,4)
    t=torch.multinomial(policy.score_t_probs,b*l*m,replacement=True,generator=generator).reshape(b,l,m)
    noise=torch.randn(b,l,m,1,4,device=states.device,dtype=states.dtype,generator=generator)
    scores=elbo_scores(policy,states,actions,t,noise,create_graph=True)
    return score_penalty(scores,method)

