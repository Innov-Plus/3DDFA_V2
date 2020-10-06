import torch.nn as nn
import torch
# from torchsummary import summary

class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 16, 2),
            conv_dw(16, 32, 1),
            conv_dw(32, 64, 1),
            conv_dw(64, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 2),
            conv_dw(64, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 1),
            conv_dw(256, 256, 2),
            conv_dw(256, 62, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(62, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 62)
        x = self.fc(x)
        return x

if __name__=='__main__':
    # model check
    model = MobileNetV1(ch_in=3, n_classes=1000)
    # summary(model, input_size=(3, 224, 224), device='cpu')

    x = torch.randn(1, 3, 120, 120, requires_grad=True)
		
    traced_model = torch.jit.trace(model, x)
    torch.jit.save(traced_model, "mobilenetV1.pt")