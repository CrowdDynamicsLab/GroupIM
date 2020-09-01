import torch


def ndcg_binary_at_k_batch_torch(X_pred, heldout_batch, k=100, device='cpu'):
    """
    Normalized Discounted Cumulative Gain@k for for predictions [B, I] and ground-truth [B, I], with binary relevance.
    ASSUMPTIONS: all the 0's in heldout_batch indicate 0 relevance.
    """

    batch_users = X_pred.shape[0]  # batch_size
    _, idx_topk = torch.topk(X_pred, k, dim=1, sorted=True)
    tp = 1. / torch.log2(torch.arange(2, k + 2, device=device).float())
    heldout_batch_nonzero = (heldout_batch > 0).float()
    DCG = (heldout_batch_nonzero[torch.arange(batch_users, device=device).unsqueeze(1), idx_topk] * tp).sum(dim=1)
    heldout_nonzero = (heldout_batch > 0).sum(dim=1)  # num. of non-zero items per batch. [B]
    IDCG = torch.tensor([(tp[:min(n, k)]).sum() for n in heldout_nonzero]).to(device)
    return DCG / IDCG


def recall_at_k_batch_torch(X_pred, heldout_batch, k=100):
    """
    Recall@k for predictions [B, I] and ground-truth [B, I].
    """
    batch_users = X_pred.shape[0]
    _, topk_indices = torch.topk(X_pred, k, dim=1, sorted=False)  # [B, K]
    X_pred_binary = torch.zeros_like(X_pred)
    if torch.cuda.is_available():
        X_pred_binary = X_pred_binary.cuda()
    X_pred_binary[torch.arange(batch_users).unsqueeze(1), topk_indices] = 1
    X_true_binary = (heldout_batch > 0).float()  # .toarray() #  [B, I]
    k_tensor = torch.tensor([k], dtype=torch.float32)
    if torch.cuda.is_available():
        X_true_binary = X_true_binary.cuda()
        k_tensor = k_tensor.cuda()
    tmp = (X_true_binary * X_pred_binary).sum(dim=1).float()
    recall = tmp / torch.min(k_tensor, X_true_binary.sum(dim=1).float())
    return recall
