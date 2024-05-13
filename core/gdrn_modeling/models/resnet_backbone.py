from collections import OrderedDict
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
import math
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck
from mmcv.cnn import normal_init, constant_init
from typing import Any, Callable, Dict, List, Optional, Tuple

from torchvision.models.regnet import BlockParams, SimpleStemIN, ResBottleneckBlock, AnyStage

# Specification
resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], "resnet18"),
    34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], "resnet34"),
    50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], "resnet50"),
    101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], "resnet101"),
    152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], "resnet152"),
}

class md_pointnet(nn.Module):

    def __init__(self, in_channel=3, conv_channels=[128, 128, 256, 512], aggregate_layer=3):
        super(md_pointnet, self).__init__()
        self.conv_channels = conv_channels
        self.aggregate_layer = aggregate_layer
        self.xyz_emb = nn.Conv2d(in_channel, conv_channels[0], kernel_size=1)
        self.xb = nn.BatchNorm2d(conv_channels[0])

        self.conv1 = nn.Conv2d(conv_channels[0]+3, conv_channels[1], kernel_size=1)
        self.conv2 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=1)
        self.conv3 = nn.Conv2d(conv_channels[2], conv_channels[3], kernel_size=1)
        self.b1 = nn.BatchNorm2d(conv_channels[1])
        self.b2 = nn.BatchNorm2d(conv_channels[2])
        self.b3 = nn.BatchNorm2d(conv_channels[3])

    def forward(self, input_ft, xyz):
        # input_ft: [3, 512, 32, 32]
        # xyz: [3, 3, 32, 32]
        b, c, h, w = xyz.size() 
        emb =F.relu(self.xb(self.xyz_emb(input_ft)), inplace=True) # [3, 64, 32, 32]
        #print("emb",emb.shape)
        xyz_e = torch.cat([xyz, emb], dim=1)

        l1 = F.relu(self.b1(self.conv1(xyz_e)), inplace=True) # [3, 128, 32, 32]
        l2 = F.relu(self.b2(self.conv2(l1)), inplace=True) # [3, 256, 32, 32]
        l3 = self.b3(self.conv3(l2)) # [3, 512, 32, 32]

        gl_ft = F.adaptive_max_pool2d(l3, (1, 1)) # [3, 512, 1, 1]
        gl_ft = F.adaptive_avg_pool2d(gl_ft, (h, w))  # [3, 512, 32, 32]

        return torch.cat([l3, gl_ft], dim=1)

class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
    

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )
        self.up_layer = nn.UpsamplingBilinear2d(scale_factor=4)
        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))
        self.spatial_net = md_pointnet(512, [64, 128, 256, 512])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
                nn.Conv2d(1512, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512),
            )

        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        xyz = x[:, 3:, :, :]
        x = x[:, :3, :, :]
        xyz = F.interpolate(xyz, (32, 32), mode='nearest')
        # torch.Size([8, 3, 256, 256])
        x = self.stem(x)
        # torch.Size([8, 32, 128, 128])
        x = self.trunk_output(x)
        # torch.Size([8, 1512, 8, 8])
        x = self.fc(x)
        # torch.Size([8, 512, 8, 8])
        x_high_feature = self.up_layer(x)
        x_high_feature = self.spatial_net(x_high_feature, xyz) 
        
        return x_high_feature

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.backbone = RegNet(block_params=BlockParams.from_init_params(depth=21, w_0=80, w_a=42.63, w_m=2.66,
                                          group_width=24, se_ratio=0.25))
    def forward(self, x):
        return self.backbone(x)
    
class MyResNetBackboneNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, freeze=False, rot_concat=False):
        self.freeze = freeze
        self.rot_concat = rot_concat
        self.inplanes = 64
        super(MyResNetBackboneNet, self).__init__()
        self.spatial_net = md_pointnet(512, [64, 128, 256, 512])
        # * new
        self.spatial_conv1 = nn.Conv2d(64+5, 64, kernel_size=1)
        self.spatial_b1 =  nn.BatchNorm2d(64)
        self.spatial_conv2 = nn.Conv2d(128 + 5, 256, kernel_size=1)
        self.spatial_b2 =  nn.BatchNorm2d(256)
        
        # * new end
        in_channel = 3
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.up_layer = nn.UpsamplingBilinear2d(scale_factor=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x.shape [32, 3, 256, 256]
        xyz = x[:, 3:, :, :]
        x = x[:, :3, :, :]
        xyz = F.interpolate(xyz, (128, 128), mode='nearest')
        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)  # x.shape [32, 64, 128, 128]
              
                x = self.bn1(x)
                x = self.relu(x)
                x_low_feature = self.maxpool(x)  # x.shape [32, 64, 64, 64]
                x_f64 = self.layer1(x_low_feature)  # x.shape [32, 256, 64, 64]
               
                x_f32 = self.layer2(x_f64)  # x.shape [32, 512, 32, 32]
              
                x_f16 = self.layer3(x_f32)  # x.shape [32, 1024, 16, 16]
                
                x_high_feature = self.layer4(x_f16)  # x.shape [32, 2048, 8, 8]
                if self.rot_concat:
                    return x_high_feature.detach(), x_f64.detach(), x_f32.detach(), x_f16.detach()
                else:
                    return x_high_feature.detach()
        else:
            x = self.conv1(x)  # x.shape [32, 64, 128, 128]
            x = self.bn1(x)
            x = self.relu(x)
            x_low_feature = self.maxpool(x)  # x.shape [32, 64, 64, 64]
            
            
           
            x_spatial = F.relu(self.spatial_b1(self.spatial_conv1(torch.cat([x, xyz], dim=1))), inplace=True) # x.shape[32, 64, 128, 128]
          
            x_spatial = F.adaptive_max_pool2d(x_spatial, (1, 1)) 
            x_spatial = F.adaptive_avg_pool2d(x_spatial, (64, 64))# x.shape[32, 64, 64, 64]
            
            
            x_f64 = self.layer1(x_low_feature + x_spatial)  # x.shape [32, 64, 64, 64]
            x_f32 = self.layer2(x_f64)  # x.shape [32, 128, 32, 32]
            xyz = F.interpolate(xyz, (32, 32), mode='nearest')
            x_spatial = F.relu(self.spatial_b2(self.spatial_conv2(torch.cat([x_f32, xyz], dim=1))), inplace=True)
            x_spatial = F.adaptive_max_pool2d(x_spatial, (1, 1)) 
            x_spatial = F.adaptive_avg_pool2d(x_spatial, (16, 16))# x.shape[32, 64, 64, 64]
            
            x_f16 = self.layer3(x_f32)  # x.shape [32, 256, 16, 16]
            x_high_feature = self.layer4(x_f16 + x_spatial)  # x.shape [32, 512, 8, 8]
            # * new add
            x_high_feature = self.up_layer(x_high_feature)  # x.shape [32, 512, 32, 32]
            #x_high_feature = self.spatial_net(x_high_feature, xyz) 
            #print('final', x_high_feature.shape)
            if self.rot_concat:
                return x_high_feature, x_f64, x_f32, x_f16
            else:
                return x_high_feature
        
        
class ResNetBackboneNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, freeze=False, rot_concat=False):
        self.freeze = freeze
        self.rot_concat = rot_concat
        self.inplanes = 64
        super(ResNetBackboneNet, self).__init__()
        self.spatial_net = md_pointnet(512, [64, 128, 256, 512])
        in_channel = 3
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.up_layer = nn.UpsamplingBilinear2d(scale_factor=4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x.shape [32, 3, 256, 256]
        xyz = x[:, 3:, :, :]
        x = x[:, :3, :, :]
        xyz = F.interpolate(xyz, (32, 32), mode='nearest')
        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)  # x.shape [32, 64, 128, 128]
              
                x = self.bn1(x)
                x = self.relu(x)
                x_low_feature = self.maxpool(x)  # x.shape [32, 64, 64, 64]
                x_f64 = self.layer1(x_low_feature)  # x.shape [32, 256, 64, 64]
               
                x_f32 = self.layer2(x_f64)  # x.shape [32, 512, 32, 32]
              
                x_f16 = self.layer3(x_f32)  # x.shape [32, 1024, 16, 16]
                
                x_high_feature = self.layer4(x_f16)  # x.shape [32, 2048, 8, 8]
                if self.rot_concat:
                    return x_high_feature.detach(), x_f64.detach(), x_f32.detach(), x_f16.detach()
                else:
                    return x_high_feature.detach()
        else:
            x = self.conv1(x)  # x.shape [32, 64, 128, 128]
            x = self.bn1(x)
            x = self.relu(x)
            x_low_feature = self.maxpool(x)  # x.shape [32, 64, 64, 64]
            x_f64 = self.layer1(x_low_feature)  # x.shape [32, 64, 64, 64]
            x_f32 = self.layer2(x_f64)  # x.shape [32, 128, 32, 32]
            x_f16 = self.layer3(x_f32)  # x.shape [32, 256, 16, 16]
            x_high_feature = self.layer4(x_f16)  # x.shape [32, 512, 8, 8]
            # * new add
            x_high_feature = self.up_layer(x_high_feature)  # x.shape [32, 512, 32, 32]
            x_high_feature = self.spatial_net(x_high_feature, xyz) 
            if self.rot_concat:
                return x_high_feature, x_f64, x_f32, x_f16
            else:
                return x_high_feature


x = torch.zeros([8,8,256,256])
block_type, layers, channels, name = resnet_spec[34]
model = MyResNetBackboneNet(block_type, layers, 3, freeze=False, rot_concat=False)


model(x)