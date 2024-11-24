import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SiameseNetwork
from torch import nn
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
# Open and transform the images
image1 = Image.open('/home/anirudhan/Documents/research/sign_data-20240127T044746Z-001/sign_data/photo_2024-01-28_07-44-04.jpg')
image2 = Image.open('/home/anirudhan/Documents/research/sign_data-20240127T044746Z-001/sign_data/photo_2024-01-28_07-43-57.jpg')

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

image1 = transform(image1)
image2 = transform(image2)
y = torch.tensor([0])
# Load the model's checkpoints
model = SiameseNetwork()
distance = ContrastiveLoss()

checkpoint = torch.load('/home/anirudhan/Documents/research/sign_data-20240127T044746Z-001/sign_data/model (2).pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

model = model.eval()
# Predict the output
output1 = model(image1.unsqueeze(0), image2.unsqueeze(0))

# Display the output value
print(output1)

dist = distance(output1[0], output1[1], y.unsqueeze(0))
print(dist)