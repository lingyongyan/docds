import torch


def masked_softmax(

    vector: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    memory_efficient: bool = False,
    mask_fill_value: float = -1e32,
) -> torch.Tensor:

    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        vector = masked_logits(vector, mask)
    return torch.nn.functional.log_softmax(vector, dim=dim)


def masked_logits(vector: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    while mask.dim() < vector.dim():
        mask = mask.unsqueeze(1)
    vector = vector + (mask + 1e-45).log()
    return vector


def sequence_mask(lens, max_len=None):
    if max_len is None:
        max_len = lens.max().item()
    batch_size = lens.size(0)

    len_range = torch.arange(max_len, device=lens.device,
                             dtype=torch.long).unsqueeze(0).expand(batch_size, max_len)
    mask = len_range < lens.unsqueeze(-1)
    return mask

def dsloss(probs, mask, candidate_mask, mask_fill_value=0,
           risk_sensitive=False, lambda_weight=0.5, gamma=2):
    if probs.size(0) <= 0:
        return 0
    tensor = - torch.log(torch.clamp(probs, min=1e-45))
    golden_mask = mask.bool().float()
    other_mask = candidate_mask - golden_mask
    na_index = torch.tensor([0], device=other_mask.device)

    # max likelihood of correct answer
    scale_probs = probs.detach()
    group_count = mask.max(dim=-1, keepdim=True)[0].clamp(min=1)
    x = torch.zeros_like(golden_mask).to(golden_mask.device)
    x = x.scatter_add(-1, mask.long(), scale_probs)
    x = x.index_fill(-1, na_index, 1)
    group_sum = x.gather(1, mask.long())
    group_prob = scale_probs / group_sum
    confidence_weight = group_prob * golden_mask
    confidence_weight = confidence_weight.detach()
    golden_scale = torch.log(group_count).detach() # constant
    golden_loss = (confidence_weight * (tensor - golden_scale)).sum(dim=-1, keepdim=True)

    # min likelihood of wrong non-NA answer
    other_mask = other_mask.index_fill(-1, na_index, 0)
    other_probs = probs * other_mask
    other_probs = other_probs
    other_scale = torch.log(torch.clamp(other_mask.sum(dim=-1, keepdim=True), min=1.)).detach() # constant
    other_loss = (other_probs * (tensor - other_scale)).sum(dim=-1, keepdim=True)

    loss = golden_loss - lambda_weight * other_loss

    # risk of samples
    if risk_sensitive:
        golden_max = torch.max(probs * golden_mask, dim=-1, keepdim=True)[0]
        pos_max = torch.max((probs * candidate_mask)[:, 1:],
                            dim=-1, keepdim=True)[0]
        other_sum = torch.sum((probs * other_mask)[:, 1:],
                              dim=-1, keepdim=True)[0]
        probs_NA = probs[:, 0:1]
        r1 = golden_max - pos_max
        r2 = torch.clamp(probs_NA - other_sum, max=0)
        risk = 1 - (r1 + r2)
        risk = torch.pow(risk, gamma)
    else:
        risk = 1.0
    loss = torch.mean(risk * loss)
    return loss


def ce_loss(tensor, mask, candidate_mask, mask_fill_value=0,
            risk_sensitive=False, gamma=2.):
    if tensor.size(0) <= 0:
        return 0
    bag = tensor * mask
    masked_bag = bag.sum(dim=-1, keepdim=True)
    if risk_sensitive:
        probs = torch.exp(-tensor)
        other_mask = candidate_mask - mask
        golden_max = torch.max(probs * mask, dim=-1, keepdim=True)[0]
        pos_max = torch.max((probs * candidate_mask)[:, 1:],
                            dim=-1, keepdim=True)[0]
        other_sum = torch.sum((probs * other_mask)[:, 1:],
                              dim=-1, keepdim=True)[0]
        probs_NA = probs[:, 0:1]
        r1 = golden_max - pos_max
        r2 = torch.clamp(probs_NA - other_sum, max=0)
        risk = 1 - torch.clamp(r1 + r2, max=0.9)
        risk = torch.pow(risk, gamma)
    else:
        risk = 1.0
    loss = torch.mean(risk * masked_bag)
    return loss


def min_ce_loss(tensor, mask, candidate_mask, mask_fill_value=float('inf')):
    if tensor.size(0) <= 0:
        return 0
    bag = tensor * mask
    masked_bag = bag.masked_fill(~mask, mask_fill_value)
    masked_bag = torch.min(masked_bag, dim=-1)[0]
    loss = torch.sum(masked_bag)
    return loss


def f_measure(prec, recall, beta=1.0):
    div = beta * beta * prec + recall
    upper = (1 + beta * beta) * prec * recall
    f1 = (float(upper) / div) if div > 0 else 0.
    return f1
