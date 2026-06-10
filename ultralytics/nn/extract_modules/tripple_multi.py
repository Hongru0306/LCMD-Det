import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1) 


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention_multi(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention_multi, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):  # x->CHW
        rgb, ir = x[0], x[1]
        rgb_perm1 = rgb.permute(0, 2, 1, 3).contiguous()  # HCW
        rgb_out1 = self.cw(rgb_perm1)
        rgb_out11 = rgb_out1.permute(0, 2, 1, 3).contiguous()
        rgb_perm2 = rgb.permute(0, 3, 2, 1).contiguous()  # WHC
        rgb_out2 = self.hc(rgb_perm2)
        rgb_out21 = rgb_out2.permute(0, 3, 2, 1).contiguous()
        rgb_out_ = self.hw(rgb)
        rgb_out = 1 / 3* (rgb_out11 + rgb_out21 + rgb_out_)
        
        ir_out = self.hw(ir)
        
        return torch.cat((rgb_out, ir_out), dim=1)


class TripletAttention_dynamic(nn.Module):  
    def __init__(self, feat_channels=512, no_spatial=False): 
        super(TripletAttention_dynamic, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()


        self.weight_generator_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(feat_channels, feat_channels // 8, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(feat_channels // 8, 3, 1), 
            nn.Softmax(dim=1) 
        )
        self.weight_generator_ir = nn.Sequential(

            nn.Conv2d(3, 1, 1), 
            nn.Sigmoid() 
        )


    def forward(self, x): 
        rgb, ir = x[0], x[1]

        # RGB Branch
        rgb_perm1 = rgb.permute(0, 2, 1, 3).contiguous()  # HCW
        rgb_out1 = self.cw(rgb_perm1)
        rgb_out11 = rgb_out1.permute(0, 2, 1, 3).contiguous()
        rgb_perm2 = rgb.permute(0, 3, 2, 1).contiguous()  # WHC
        rgb_out2 = self.hc(rgb_perm2)
        rgb_out21 = rgb_out2.permute(0, 3, 2, 1).contiguous()  
        rgb_out_hw = self.hw(rgb) 


        dynamic_weights_rgb = self.weight_generator_rgb(rgb) 
        w_cw_rgb, w_hc_rgb, w_hw_rgb = dynamic_weights_rgb.split(1, dim=1)


        rgb_out_combined = w_cw_rgb * rgb_out11 + w_hc_rgb * rgb_out21 + w_hw_rgb * rgb_out_hw 


        ir_out_hw = self.hw(ir)
        dynamic_weight_ir_hw = self.weight_generator_ir(dynamic_weights_rgb) 
        ir_out_combined = dynamic_weight_ir_hw * ir_out_hw 


        return torch.cat((rgb_out_combined, ir_out_combined), dim=1)




if __name__ == '__main__':
    rgb_input = torch.randn(8, 512, 14, 14)  # Example RGB feature map
    ir_input = torch.randn(8, 512, 14, 14)   # Example IR feature map

    triplet = TripletAttention_dynamic(512)
    output = triplet([rgb_input, ir_input])
    print(output.shape)