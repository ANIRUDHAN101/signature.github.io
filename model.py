from torch import nn
import torch
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    "Contrastive loss function"
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
            
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2,keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

from torchvision.models import efficientnet_v2_s

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = efficientnet_v2_s(weights="DEFAULT")
        self.feature_extractor.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.feature_extractor.classifier = nn.Identity()
        self.embedding = nn.Sequential(
            nn.Linear(1280, 2),
            # nn.GELU(),
            # nn.Linear(128, 2)
        )

    def forward(self, x1, x2):
        output1 = self.feature_extractor(x1)
        output2 = self.feature_extractor(x2)

        output1 = output1.view(output1.size()[0], -1)
        output2 = output2.view(output2.size()[0], -1)

        output1 = self.embedding(output1)
        output2 = self.embedding(output2)

        return output1, output2
