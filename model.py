from lib import *

class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.model = self.get_model()
    def get_model(self):
        model = torchvision.models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 10, bias=True)
        return model
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = torchvision.models.resnet18()
    print(model)
    cifar = CIFARModel()
    print(cifar)