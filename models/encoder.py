import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """ User Preference Encoder implemented as fully connected layers over binary bag-of-words vector
    (over item set) per user """

    def __init__(self, n_items, user_layers, embedding_dim, drop_ratio):
        super(Encoder, self).__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.drop = nn.Dropout(drop_ratio)
        self.user_preference_encoder = torch.nn.ModuleList()  # user individual preference encoder layers.

        for idx, (in_size, out_size) in enumerate(zip([self.n_items] + user_layers[:-1], user_layers)):
            layer = torch.nn.Linear(in_size, out_size, bias=True)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.user_preference_encoder.append(layer)

        self.transform_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_uniform_(self.transform_layer.weight)
        nn.init.zeros_(self.transform_layer.bias)

        self.user_predictor = nn.Linear(self.embedding_dim, self.n_items, bias=False)  # item embedding for pre-training
        nn.init.xavier_uniform_(self.user_predictor.weight)

    def pre_train_forward(self, user_items):
        """ user individual preference encoder (excluding final layer) for user-item pre-training
            :param user_items: [B, G, I] or [B, I]
        """
        user_items_norm = F.normalize(user_items)  # [B, G, I] or [B, I]
        user_pref_embedding = self.drop(user_items_norm)
        for idx, _ in enumerate(range(len(self.user_preference_encoder))):
            user_pref_embedding = self.user_preference_encoder[idx](user_pref_embedding)  # [B, G, D] or [B, D]
            user_pref_embedding = torch.tanh(user_pref_embedding)  # [B, G, D] or [B, D]

        logits = self.user_predictor(user_pref_embedding)  # [B, G, D] or [B, D]
        return logits, user_pref_embedding

    def forward(self, user_items):
        """ user individual preference encoder
            :param user_items: [B, G, I]
        """
        _, user_embeds = self.pre_train_forward(user_items)  # [B, G, D]
        user_embeds = torch.tanh(self.transform_layer(user_embeds))  # [B, G, D]
        return user_embeds
