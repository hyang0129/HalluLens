


from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def evaluate(model, test_dataloader, batch_size=32, loss_fn=None, device='cuda', sub_batch_size=64):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_dataloader:
            x1 = batch['layer1_activations'].squeeze(1)
            x2 = batch['layer2_activations'].squeeze(1)

            batch_size_actual = x1.size(0)
            batch_loss = 0.0
            batch_acc = 0.0
            num_sub_batches = 0

            for i in range(0, batch_size_actual, sub_batch_size):
                x1_chunk = x1[i:i+sub_batch_size].to(device, non_blocking=True)
                x2_chunk = x2[i:i+sub_batch_size].to(device, non_blocking=True)

                z1 = model(x1_chunk)
                z2 = model(x2_chunk)

                z_stacked = torch.stack([z1, z2], dim=1)

                loss = loss_fn(z_stacked)
                acc = pairing_accuracy(z1, z2)

                batch_loss += loss.item()
                batch_acc += acc
                num_sub_batches += 1

            total_loss += batch_loss / num_sub_batches
            total_acc += batch_acc / num_sub_batches
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    return avg_loss, avg_acc

def train_contrastive(model, train_dataset, test_dataset=None,
                      epochs=10, batch_size=512, lr=1e-6,
                      temperature=0.07, device='cuda', num_workers=16, sub_batch_size=64):
    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = SupConLoss(temperature=temperature)

    train_loader = DataLoader(
        train_dataset,
        batch_size=sub_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    subsinbatch = batch_size // sub_batch_size
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        # Buffers to accumulate mini-batches
        buffer_x1, buffer_x2 = [], []

        i = 0
        for batch in loop:
            i += 1 
            x1 = batch['layer1_activations'].squeeze(1).to(device, non_blocking=True)
            x2 = batch['layer2_activations'].squeeze(1).to(device, non_blocking=True)

            buffer_x1.append(x1)
            buffer_x2.append(x2)

            # Once we accumulate enough to reach full batch size
            if len(buffer_x1) * sub_batch_size == batch_size:
                x1_full = torch.cat(buffer_x1, dim=0)
                x2_full = torch.cat(buffer_x2, dim=0)
                # print(len(buffer_x1))
                buffer_x1 = [] 
                buffer_x2 = [] 

                z1 = model(x1_full)
                z2 = model(x2_full)

                z_stacked = torch.stack([z1, z2], dim=1)
                loss = loss_fn(z_stacked)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = pairing_accuracy(z1, z2)
                total_loss += loss.item()
                total_acc += acc

                # print(batch_size // sub_batch_size)
                # print(sub_batch_size)
                # print(loop.n)
                
                avg_loss = total_loss / (i / subsinbatch)
                avg_acc = total_acc / ( i / subsinbatch)

                loop.set_postfix(loss=avg_loss, pairing_acc=avg_acc)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f} - Train Pairing Acc: {avg_acc:.4f}")

        if test_dataset is not None:
            test_loss, test_acc = evaluate(model, test_dataset, batch_size=batch_size, loss_fn=loss_fn, device=device, sub_batch_size= sub_batch_size)
            print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss:.4f} - Test Pairing Acc: {test_acc:.4f}")


def pairing_accuracy(z1, z2):
    """
    z1, z2: (B, D) normalized embeddings
    Returns percentage of pairs correctly matched by highest cosine similarity.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute cosine similarity matrix (B x B)
    sim_matrix = torch.matmul(z1, z2.T)  # (B, B)

    # For each in z1, find index of max similarity in z2
    preds_z1 = sim_matrix.argmax(dim=1)
    # For each in z2, find index of max similarity in z1
    preds_z2 = sim_matrix.argmax(dim=0)

    batch_size = z1.size(0)
    # Correct matches: indices where preds match i
    correct_z1 = (preds_z1 == torch.arange(batch_size, device=z1.device)).sum().item()
    correct_z2 = (preds_z2 == torch.arange(batch_size, device=z2.device)).sum().item()

    # Average accuracy
    accuracy = (correct_z1 + correct_z2) / (2 * batch_size)
    return accuracy

def inference_embeddings(model, dataset, batch_size=512, sub_batch_size=64, device='cuda', num_workers=16):
    assert batch_size % sub_batch_size == 0, "batch_size must be divisible by sub_batch_size"

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=sub_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    buffer_x1, buffer_x2, buffer_hash = [], [], []
    results = []

    subs_in_batch = batch_size // sub_batch_size

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            x1 = batch['layer1_activations'].squeeze(1).to(device, non_blocking=True)
            x2 = batch['layer2_activations'].squeeze(1).to(device, non_blocking=True)
            hashkeys = batch['hashkey']

            buffer_x1.append(x1)
            buffer_x2.append(x2)
            buffer_hash.extend(hashkeys)

            if len(buffer_x1) == subs_in_batch:
                x1_full = torch.cat(buffer_x1, dim=0)
                x2_full = torch.cat(buffer_x2, dim=0)

                buffer_x1, buffer_x2 = [], []

                z1 = model(x1_full)
                z2 = model(x2_full)

                for h, z1_i, z2_i in zip(buffer_hash, z1, z2):
                    results.append({
                        "hashkey": h,
                        "z1": z1_i.cpu(),
                        "z2": z2_i.cpu()
                    })

                buffer_hash = []

    return results
