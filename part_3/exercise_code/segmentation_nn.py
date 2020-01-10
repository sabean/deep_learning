"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        """
        self.vgg16_bn = models.vgg16_bn(pretrained=True)
        
        for param in self.vgg16_bn.parameters():
            param.requires_grad = False
        
        features = list(self.vgg16_bn.features)
        self.vgg16_features = nn.ModuleList(features).eval()
        
        self.relu    = nn.ReLU(inplace=True)
        self.deconvolution_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.batch_norm_1     = nn.BatchNorm2d(512)
        
        self.deconvolution_2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.batch_norm_2     = nn.BatchNorm2d(256)
        
        self.deconvolution_3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.batch_norm_3     = nn.BatchNorm2d(128)
        
        self.deconvolution_4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.batch_norm_4     = nn.BatchNorm2d(64)
        
        self.deconvolution_5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.batch_norm_5     = nn.BatchNorm2d(32)
        
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        self.vgg16_bn = models.vgg11_bn(pretrained=True).features
        self.network = nn.Sequential(
            nn.Conv2d(512,256,5),
            nn.ReLU(),
            nn.Conv2d(256,128,1)        
        )"""
        self.num_classes = num_classes
        self.vgg_feat = models.vgg11(pretrained=True).features
        self.fcn = nn.Sequential(
                                nn.Conv2d(512, 1024, 7),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Conv2d(1024, 2048, 1),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Conv2d(2048, num_classes, 1)
                                )
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        """feature_outputs = {}
        selection = [6, 13, 23, 33, 43]
        
        count = 1
        for i, model in enumerate(self.vgg16_features):
            x = model(x)
            if i in selection:
                #feature_outputs.append(x)
                feature_outputs[count] = x
                count += 1
        

        score = self.batch_norm_1(self.relu(self.deconvolution_1(feature_outputs[5])))
        score += feature_outputs[4]  
        
        score = self.batch_norm_2(self.relu(self.deconvolution_2(score)))
        score += feature_outputs[3]
        
        score = self.batch_norm_3(self.relu(self.deconvolution_3(score)))
        score += feature_outputs[2]
        
        score = self.batch_norm_4(self.relu(self.deconvolution_4(score)))
        score += feature_outputs[1]
        
        score = self.batch_norm_(self.relu(self.deconvolution_5(score)))
        score = self.classifier(score)
        
        x = score
        
        x_shape = x.shape
        
        x = self.vgg16_bn(x)
        x = self.network(x)
        x = F.upsample(x,x_shape[2:])"""
        x_input = x
        x = self.vgg_feat(x)
        x = self.fcn(x)
        x = F.upsample(x, x_input.size()[2:], mode='bilinear', align_corners=True).contiguous()
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
