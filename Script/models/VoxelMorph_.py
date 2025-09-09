import torch
import torch.nn as nn

from models.base_block import SpatialTransformer, SpatialTransformer1, SpatialTransformer2, BaseModule
from models.UNet import UNet


class VoxelMorph(BaseModule):
    def __init__(self,
                 backbone,
                 feat_num,
                 img_size,
                 integrate_cfg,
                 ):
        super(VoxelMorph, self).__init__()
        self.backbone = backbone

        self.feat_num = feat_num

        self.img_size = img_size

        if self.backbone == 'UNet':
            self.feat_extractor = UNet(self.feat_num)
        else:
            raise NotImplementedError

        self.flow = nn.Conv2d(self.feat_num[0], 2, kernel_size=3, padding=1)

        self.init_weight()
        # init flow layer with small weights and bias
        nn.init.normal_(self.flow.weight, mean=0, std=1e-5)
        nn.init.constant_(self.flow.bias, 0)

        self.stn = SpatialTransformer1(self.img_size)  

        self.integrate_cfg = integrate_cfg

        self.down_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        if self.integrate_cfg['UseIntegrate']:
            self.vec_int = VecInt(self.img_size, integrate_cfg['TimeStep'])

    def forward(self, moving, fixed):

        fwd = {'Moving': moving, 'Fixed': fixed}

        size = fwd['Moving'].size()[0]
        height = fwd['Moving'].size()[2]
        width = fwd['Moving'].size()[3]
        init_type = fwd['Moving'].dtype
        init_device = fwd['Moving'].device
        init_flow = torch.zeros(size, 2, int(height / 4), int(width / 4), dtype=init_type, device=init_device).float()  

        x = torch.cat([moving, fixed], dim=1)
        cat_input = self.down_avg(x)
        cat_input1 = self.down_avg(cat_input)
        down_x = cat_input1[:, 0:1, :, :]
        down_y = cat_input1[:, 1:2, :, :]

        for i in range(5):
            input_1 = torch.cat((down_x, down_y), 1)
            feature = self.feat_extractor(input_1)
            if self.integrate_cfg['UseIntegrate']:
                velocity = self.flow(feature)
                flow_filed = self.vec_int(velocity)
                fwd['Velocity'] = velocity
            else:
                flow_filed = self.flow(feature)
            init_flow = init_flow + flow_filed
            y = self.stn(down_x, init_flow)
            down_x = y

        fwd['Flow'] = init_flow

        # moved = self.stn(moving, flow_filed) 
        fwd['Moved'] = down_x
        fwd['Fixed'] = down_y
        return fwd


class VoxelMorph2(BaseModule):
    def __init__(self,
                 backbone,
                 feat_num,
                 img_size,
                 integrate_cfg,
                 model1
                 ):
        super(VoxelMorph2, self).__init__()
        self.backbone = backbone

        self.feat_num = feat_num

        self.img_size = img_size

        if self.backbone == 'UNet':
            self.feat_extractor = UNet(self.feat_num)
        else:
            raise NotImplementedError

        self.flow = nn.Conv2d(self.feat_num[0], 2, kernel_size=3, padding=1)

        self.init_weight()
        # init flow layer with small weights and bias
        nn.init.normal_(self.flow.weight, mean=0, std=1e-5)
        nn.init.constant_(self.flow.bias, 0)

        self.stn = SpatialTransformer2(self.img_size) 

        self.integrate_cfg = integrate_cfg

        self.down_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.model1 = model1
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        if self.integrate_cfg['UseIntegrate']:
            self.vec_int = VecInt(self.img_size, integrate_cfg['TimeStep'])

    def forward(self, moving, fixed):

        fwd = {'Moving': moving, 'Fixed': fixed}

        size = fwd['Moving'].size()[0]
        height = fwd['Moving'].size()[2]
        width = fwd['Moving'].size()[3]
        init_type = fwd['Moving'].dtype
        init_device = fwd['Moving'].device
        init_flow = torch.zeros(size, 2, int(height / 2), int(width / 2), dtype=init_type, device=init_device).float()  

        fwd1 = self.model1(moving, fixed)
        Upflow1 = self.up(fwd1['Flow'])
        x_down = self.down_avg(moving)
        down_y = self.down_avg(fixed)
        down_x = self.stn(x_down, Upflow1)

        for i in range(5):
            input_1 = torch.cat((down_x, down_y), 1)
            feature = self.feat_extractor(input_1)
            if self.integrate_cfg['UseIntegrate']:
                velocity = self.flow(feature)
                flow_filed = self.vec_int(velocity)
                fwd['Velocity'] = velocity
            else:
                flow_filed = self.flow(feature)
            init_flow = init_flow + flow_filed
            y = self.stn(down_x, init_flow)
            down_x = y

        fwd['Flow'] = init_flow

        # moved = self.stn(moving, flow_filed) 

        fwd['Moved'] = down_x
        fwd['Fixed'] = down_y
        return fwd


class VoxelMorph3(BaseModule):
    def __init__(self,
                 backbone,
                 feat_num,
                 img_size,
                 integrate_cfg,
                 model1,
                 model2
                 ):
        super(VoxelMorph3, self).__init__()
        self.backbone = backbone

        self.feat_num = feat_num

        self.img_size = img_size

        if self.backbone == 'UNet':
            self.feat_extractor = UNet(self.feat_num)
        else:
            raise NotImplementedError

        self.flow = nn.Conv2d(self.feat_num[0], 2, kernel_size=3, padding=1)

        self.init_weight()
        # init flow layer with small weights and bias
        nn.init.normal_(self.flow.weight, mean=0, std=1e-5)
        nn.init.constant_(self.flow.bias, 0)

        self.stn = SpatialTransformer(self.img_size) 

        self.integrate_cfg = integrate_cfg

        self.down_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.model1 = model1
        self.model2 = model2
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear')

        if self.integrate_cfg['UseIntegrate']:
            self.vec_int = VecInt(self.img_size, integrate_cfg['TimeStep'])

    def forward(self, moving, fixed):

        fwd = {'Moving': moving, 'Fixed': fixed}

        size = fwd['Moving'].size()[0]
        height = fwd['Moving'].size()[2]
        width = fwd['Moving'].size()[3]
        init_type = fwd['Moving'].dtype
        init_device = fwd['Moving'].device
        init_flow = torch.zeros(size, 2, int(height), int(width), dtype=init_type, device=init_device).float() 

        fwd1 = self.model1(moving, fixed)
        fwd2 = self.model2(moving, fixed)
        upflow1 = self.up(fwd1['Flow'])
        Upflow1 = self.up(upflow1)
        upflow2 = self.up(fwd2['Flow'])
        Upflow2 = Upflow1 + upflow2

        down_x = self.stn(moving, Upflow2)

        for i in range(5):
            input_1 = torch.cat((down_x, fixed), 1)
            feature = self.feat_extractor(input_1)
            if self.integrate_cfg['UseIntegrate']:
                velocity = self.flow(feature)
                flow_filed = self.vec_int(velocity)
                fwd['Velocity'] = velocity
            else:
                flow_filed = self.flow(feature)
            init_flow = init_flow + flow_filed
            y = self.stn(down_x, init_flow)
            down_x = y

        fwd['Flow'] = init_flow

        # moved = self.stn(moving, flow_filed)  

        fwd['Moved'] = down_x
        return fwd




class VoxelMorph22(BaseModule):
    def __init__(self,
                 backbone,
                 feat_num,
                 img_size,
                 integrate_cfg,
                 model1,
                 model2,
                 model3
                 ):
        super(VoxelMorph22, self).__init__()
        self.backbone = backbone

        self.feat_num = feat_num

        self.img_size = img_size

        if self.backbone == 'UNet':
            self.feat_extractor = UNet(self.feat_num)
        else:
            raise NotImplementedError

        self.flow = nn.Conv2d(self.feat_num[0], 2, kernel_size=3, padding=1)

        self.init_weight()
        # init flow layer with small weights and bias
        nn.init.normal_(self.flow.weight, mean=0, std=1e-5)
        nn.init.constant_(self.flow.bias, 0)

        self.stn = SpatialTransformer2(self.img_size)  

        self.integrate_cfg = integrate_cfg

        self.down_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False) 
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear') 

        if self.integrate_cfg['UseIntegrate']:
            self.vec_int = VecInt(self.img_size, integrate_cfg['TimeStep'])

    def forward(self, moving, fixed):

        fwd = {'Moving': moving, 'Fixed': fixed}

        size = fwd['Moving'].size()[0]
        height = fwd['Moving'].size()[2]
        width = fwd['Moving'].size()[3]
        init_type = fwd['Moving'].dtype
        init_device = fwd['Moving'].device
        init_flow = torch.zeros(size, 2, int(height / 2), int(width / 2), dtype=init_type, device=init_device).float()  

        fwd1 = self.model1(moving, fixed)
        fwd2 = self.model2(moving, fixed)
        fwd3 = self.model3(moving, fixed)

        upflow1 = self.up(fwd1['Flow'])
        Upflow1 = self.up(upflow1)
        upflow2 = self.up(fwd2['Flow'])
        Upflow2 = Upflow1 + upflow2
        Upflow3 = Upflow2 + fwd3['Flow']

        Doflow2 = self.down_avg(Upflow3)
        x_down2 = self.down_avg(moving)
        down_y2 = self.down_avg(fixed)
        down_x2 = self.stn(x_down2, Doflow2)

        # down_x = self.stn(moving, Upflow3) 

        for i in range(5):
            input_1 = torch.cat((down_x2, down_y2), 1)
            feature = self.feat_extractor(input_1)
            if self.integrate_cfg['UseIntegrate']:
                velocity = self.flow(feature)
                flow_filed = self.vec_int(velocity)
                fwd['Velocity'] = velocity
            else:
                flow_filed = self.flow(feature)
            init_flow = init_flow + flow_filed
            y = self.stn(down_x2, init_flow)
            down_x2 = y

        fwd['Flow'] = init_flow

        # moved = self.stn(moving, flow_filed)

        fwd['Moved'] = down_x2
        fwd['Fixed'] = down_y2
        return fwd


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, img_size, n_steps):
        super().__init__()

        assert n_steps >= 0, 'n_steps should be >= 0, found: %d' % n_steps
        self.n_steps = n_steps
        self.scale = 1.0 / (2 ** self.n_steps)
        # self.transformer = SpatialTransformer1((img_size, img_size))
        self.transformer = SpatialTransformer1(img_size)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.n_steps):
            vec = vec + self.transformer(vec, vec)
        return vec
