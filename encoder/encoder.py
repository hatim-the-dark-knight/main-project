import torch
import torch.nn as nn
from torchvision import models, transforms
import PIL.Image as Image

c_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResizeImage(object):
    def __init__(self, img_size):
        if img_size is None or img_size < 1:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])

    def __call__(self, img):
        img = self.transform(img)
        return img

# resize_image = ResizeImage(img_size=256)
# img = Image.open('image.jpg')
# tensor = resize_image(img)
# print(tensor)

# class ConvEncoder(nn.Module):
    
#     def __init__(self, c_dim=128):
#         super().__init__()
#         self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
#         self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
#         self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
#         self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
#         self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
#         self.fc_out = nn.Linear(512, c_dim)
#         self.actvn = nn.ReLU()

#     def forward(self, x):
#         batch_size = x.size(0)

#         net = self.conv0(x)
#         net = self.conv1(self.actvn(net))
#         net = self.conv2(self.actvn(net))
#         net = self.conv3(self.actvn(net))
#         net = self.conv4(self.actvn(net))
#         net = net.view(batch_size, 512, -1).mean(2)
#         out = self.fc_out(self.actvn(net))

#         return out


class Resnet18(nn.Module):
    
    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18()
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


def normalize_imagenet(x):
    
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225

    print(x)
    return x

encoder = Resnet18(c_dim=c_dim)

img_path = "image.jpg"
img = Image.open(img_path)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
img_tensor = transform(img)

encoder = Resnet18(c_dim=512)

output = encoder(img_tensor.unsqueeze(0))

print(output, output.shape)
