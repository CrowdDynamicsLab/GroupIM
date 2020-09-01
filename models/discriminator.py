import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """ Discriminator for Mutual Information Estimation and Maximization, implemented with bilinear layers and
    binary cross-entropy loss training """

    def __init__(self, embedding_dim=64):
        super(Discriminator, self).__init__()
        self.embedding_dim = embedding_dim

        self.fc_layer = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.zeros_(self.fc_layer.bias)

        self.bilinear_layer = nn.Bilinear(self.embedding_dim, self.embedding_dim, 1)  # output_dim = 1 => single score.
        nn.init.zeros_(self.bilinear_layer.weight)
        nn.init.zeros_(self.bilinear_layer.bias)

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, group_inputs, user_inputs, group_mask):
        """ bilinear discriminator:
            :param group_inputs: [B, I]
            :param user_inputs: [B, n_samples, I] where n_samples is either G or # negs
            :param group_mask: [B, G]
        """
        # FC + activation.
        group_encoded = self.fc_layer(group_inputs)  # [B, D]
        group_embed = torch.tanh(group_encoded)  # [B, D]

        # FC + activation.
        user_pref_embedding = self.fc_layer(user_inputs)
        user_embed = torch.tanh(user_pref_embedding)  # [B, n_samples, D]

        return self.bilinear_layer(user_embed, group_embed.unsqueeze(1).repeat(1, user_inputs.shape[1], 1))

    def mi_loss(self, scores_group, group_mask, scores_corrupted, device='cpu'):
        """ binary cross-entropy loss over (group, user) pairs for discriminator training
            :param scores_group: [B, G]
            :param group_mask: [B, G]
            :param scores_corrupted: [B, N]
            :param device (cpu/gpu)
         """
        batch_size = scores_group.shape[0]
        pos_size, neg_size = scores_group.shape[1], scores_corrupted.shape[1]

        one_labels = torch.ones(batch_size, pos_size).to(device)  # [B, G]
        zero_labels = torch.zeros(batch_size, neg_size).to(device)  # [B, N]

        labels = torch.cat((one_labels, zero_labels), 1)  # [B, G+N]
        logits = torch.cat((scores_group, scores_corrupted), 1).squeeze(2)  # [B, G + N]

        mask = torch.cat((torch.exp(group_mask), torch.ones([batch_size, neg_size]).to(device)),
                         1)  # torch.exp(.) to binarize since original mask has -inf.

        mi_loss = self.bce_loss(logits * mask, labels * mask) * (batch_size * (pos_size + neg_size)) \
                  / (torch.exp(group_mask).sum() + batch_size * neg_size)

        return mi_loss
