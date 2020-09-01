import torch
import numpy as np
from eval import metrics
import gc


def evaluate_user(model, eval_loader, device, mode='pretrain'):
    """ evaluate model on recommending items to users (primarily during pre-training step) """
    model.eval()
    eval_loss = 0.0
    n100_list, r20_list, r50_list = [], [], []
    eval_preds = []
    with torch.no_grad():
        for batch_index, eval_data in enumerate(eval_loader):
            eval_data = [x.to(device, non_blocking=True) for x in eval_data]
            (users, fold_in_items, held_out_items) = eval_data
            fold_in_items = fold_in_items.to(device)
            if mode == 'pretrain':
                recon_batch, emb = model.user_preference_encoder.pre_train_forward(fold_in_items)
            else:
                recon_batch = model.group_predictor(model.user_preference_encoder(fold_in_items))

            loss = model.multinomial_loss(recon_batch, held_out_items)
            eval_loss += loss.item()
            fold_in_items = fold_in_items.cpu().numpy()
            recon_batch = torch.softmax(recon_batch, 1)  # softmax over the item set to get normalized scores.
            recon_batch[fold_in_items.nonzero()] = -np.inf

            n100 = metrics.ndcg_binary_at_k_batch_torch(recon_batch, held_out_items, 100, device=device)
            r20 = metrics.recall_at_k_batch_torch(recon_batch, held_out_items, 20)
            r50 = metrics.recall_at_k_batch_torch(recon_batch, held_out_items, 50)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

            eval_preds.append(recon_batch.cpu().numpy())
            del users, fold_in_items, held_out_items, recon_batch
    gc.collect()
    num_batches = max(1, len(eval_loader.dataset) / eval_loader.batch_size)
    eval_loss /= num_batches
    n100_list = torch.cat(n100_list)
    r20_list = torch.cat(r20_list)
    r50_list = torch.cat(r50_list)
    return eval_loss, torch.mean(n100_list), torch.mean(r20_list), torch.mean(r50_list), np.array(eval_preds)


def evaluate_group(model, eval_group_loader, device):
    """ evaluate model on recommending items to groups """
    model.eval()
    eval_loss = 0.0
    n100_list, r20_list, r50_list = [], [], []
    eval_preds = []

    with torch.no_grad():
        for batch_idx, data in enumerate(eval_group_loader):
            data = [x.to(device, non_blocking=True) for x in data]
            group, group_users, group_mask, group_items, user_items = data
            recon_batch, _, _ = model(group, group_users, group_mask, user_items)

            loss = model.multinomial_loss(recon_batch, group_items)
            eval_loss += loss.item()
            result = recon_batch.softmax(1)  # softmax over the item set to get normalized scores.
            heldout_data = group_items

            r20 = metrics.recall_at_k_batch_torch(result, heldout_data, 20)
            r50 = metrics.recall_at_k_batch_torch(result, heldout_data, 50)
            n100 = metrics.ndcg_binary_at_k_batch_torch(result, heldout_data, 100, device=device)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

            eval_preds.append(recon_batch.cpu().numpy())
            del group, group_users, group_mask, group_items, user_items
    gc.collect()

    n100_list = torch.cat(n100_list)
    r20_list = torch.cat(r20_list)
    r50_list = torch.cat(r50_list)
    return eval_loss, torch.mean(n100_list), torch.mean(r20_list), torch.mean(r50_list), np.array(eval_preds)

