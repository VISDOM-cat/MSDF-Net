class MultiNegativeTripletLoss(nn.Module):
    def __init__(self, margin=0.5, hard_ratio=0.3):
        super().__init__()
        self.margin = margin
        self.hard_ratio = hard_ratio

    def forward(self, anchor_features, positive_features, negative_features_list):
        num_patches = anchor_features.shape[1]
        total_loss = 0.0

        for k in range(num_patches):
            dist_pos = torch.norm(
                anchor_features[:, k, :] - positive_features[:, k, :],
                p=2, dim=1
            )

            all_neg_dists = []
            for neg_feats in negative_features_list:
                dist_neg = torch.norm(
                    anchor_features[:, k, :] - neg_feats[:, k, :],
                    p=2, dim=1
                )
                all_neg_dists.append(dist_neg)

            all_neg_dists = torch.stack(all_neg_dists)

            num_all_neg = all_neg_dists.shape[0]
            num_hard = max(1, int(num_all_neg * self.hard_ratio))
            hard_neg_dists, _ = torch.topk(all_neg_dists, num_hard, dim=0, largest=False)
            hard_neg_min, _ = torch.min(hard_neg_dists, dim=0)
            loss_k = F.relu(dist_pos - hard_neg_min + self.margin).mean()
            total_loss += loss_k

        return total_loss / num_patches