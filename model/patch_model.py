import torch
import torch.nn as nn

class SemanticPatchFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128,
                 num_horizontal_patches=3, num_vertical_patches=3):
        super().__init__()
        self.num_horizontal_patches = num_horizontal_patches
        self.num_vertical_patches = num_vertical_patches
        self.total_patches = num_horizontal_patches + num_vertical_patches
        self.feature_dim = feature_dim

        self.patch_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.horizontal_1d_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, feature_dim, kernel_size=1)
        )

        self.vertical_1d_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, feature_dim, kernel_size=1)
        )

        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim)
        )

        self.img_h, self.img_w = 30, 30
        self.horizontal_patch_h = self.img_h // self.num_horizontal_patches
        self.vertical_patch_w = self.img_w // self.num_vertical_patches

    def extract_semantic_patches(self, x):
        batch_size = x.shape[0]
        patches = []
        for i in range(self.num_horizontal_patches):
            start_h = i * self.horizontal_patch_h
            end_h = start_h + self.horizontal_patch_h
            horizontal_patch = x[:, :, start_h:end_h, :]
            patches.append(horizontal_patch)

        for i in range(self.num_vertical_patches):
            start_w = i * self.vertical_patch_w
            end_w = start_w + self.vertical_patch_w
            vertical_patch = x[:, :, :, start_w:end_w]
            patches.append(vertical_patch)

        return patches

    def forward(self, x):
        batch_size = x.shape[0]
        semantic_patches = self.extract_semantic_patches(x)

        patch_feats = []
        for patch in semantic_patches:
            feat = self.patch_extractor(patch)
            patch_feats.append(feat)

        horizontal_feats = torch.stack(
            patch_feats[:self.num_horizontal_patches], dim=2
        )
        vertical_feats = torch.stack(
            patch_feats[self.num_horizontal_patches:], dim=2
        )

        horizontal_interactive = self.horizontal_1d_conv(horizontal_feats)
        vertical_interactive = self.vertical_1d_conv(vertical_feats)

        horizontal_interactive = horizontal_interactive.permute(0, 2, 1)
        vertical_interactive = vertical_interactive.permute(0, 2, 1)

        num_h_patches = horizontal_interactive.shape[1]
        num_v_patches = vertical_interactive.shape[1]

        fused_horizontal = self.final_fusion(torch.cat(
            [horizontal_interactive,
             vertical_interactive[:, :num_h_patches, :]],
            dim=2
        ))

        fused_vertical = self.final_fusion(torch.cat(
            [vertical_interactive,
             horizontal_interactive[:, :num_v_patches, :]],
            dim=2
        ))

        final_feats = torch.cat([fused_horizontal, fused_vertical], dim=1)
        return final_feats