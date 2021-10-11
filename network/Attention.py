from os import stat
from typing import Tuple
from types import MethodType
import torch
from torch.nn import init
import torch.nn.functional as F
from torch import nn
import warnings
import segmentation_models_pytorch as smp
from einops import rearrange
from network.CLSTM import New_BDCLSTM, Temp_New_BDCLSTM, CLSTM

class DeepLabV3Plus_SA(nn.Module):
    def __init__(self, num_classes,
        hidden_channels: Tuple[int, int] = (3, 8),
        continue_num = 8, backbone = "resnet34"
        ):

        super().__init__()
        aspp_channel = 128
        self.unet1 = smp.DeepLabV3Plus(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            decoder_channels=aspp_channel,
            classes=1, # model output channels (number of classes in your dataset),
        )
        self.unet1.forward = MethodType(self.deeplab_forward, self.unet1)

        self.len = continue_num
        # self.lstm = New_BDCLSTM(length = continue_num, input_channels = hidden_channels[0], hidden_channels=[hidden_channels[1]])
        self.lstm = BDCLSTM_Up(length = continue_num, input_channels = aspp_channel, hidden_channels=[aspp_channel])
    
    @staticmethod
    def deeplab_forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features) # (b, 264, 88, 104)
        return decoder_output

    def forward(self, input, other_frame):
        # print(input.shape)
        b = other_frame.size(0)
        other_frame = rearrange(other_frame, 'b t c h w -> (b t) c h w')
        continue_list = self.unet1(other_frame)
        continue_list = rearrange(continue_list, '(b t) c h w -> b t c h w', b=b)
        

        final_predict, temporal_mask = self.lstm(continue_list)
        return temporal_mask, final_predict

    
class BDCLSTM_Up(nn.Module):
        # Constructor
    def __init__(self, length, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True, num_classes=1):

        super(BDCLSTM_Up, self).__init__()
        self.len = length
        self.forward_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)

        def get_conv2x_layers(in_ch, out_ch, kernel_size=3):
            half_kernel = kernel_size//2
            return nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(hidden_channels[-1], hidden_channels[-1]//2, kernel_size=kernel_size, padding=half_kernel, stride=half_kernel),
                nn.ReLU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
            )
        
        self.feat_upconv = nn.ModuleList([get_conv2x_layers(hidden_channels[-1], num_classes, 3) for i in range(self.len)])
        self.sup_out = nn.Conv2d(hidden_channels[-1]//2, out_channels=num_classes, kernel_size=1)
        # self.conv = get_conv4x_layers(hidden_channels[-1], num_classes, 3)
        
        # self.final_conv = nn.Conv2d(self.len, num_classes, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels[-1]//2*self.len, out_channels=hidden_channels[-1]//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[-1]//2, out_channels=num_classes, kernel_size=1),
        )
        # self.final_conv = nn.Conv3d(self.len*hidden_channels[-1]//2, num_classes, kernel_size=1)
        
    def forward(self, continue_list):
        yforward = self.forward_net(continue_list)
        out_sup = []
        feat_sup = []
        for i in range(self.len):
            feat = self.feat_upconv[i](yforward[i])
            F_y = self.sup_out(feat)
            out_sup.append(F_y)
            feat_sup.append(feat)
            # total_y = torch.cat( (total_y, F_y), dim = 1)
        out_sup = torch.cat(out_sup, dim=1)
        feat_sup = torch.cat(feat_sup, dim=1)
        current_y = self.final_conv(feat_sup)
        # current_y = self.final_conv(total_y.unsqueeze(dim = 2)).squeeze(dim = 1)
        return current_y, out_sup

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = self.proj(x.reshape(-1, c, h, w)).flatten(2).transpose(1, 2)
        # (b t c h w) -> (b*t c h w) -> (b*t emb h' w') -> (b, t*h'*w', emb)
        return x

class Attention(nn.Module):
    # timm
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # x: (B, N, C)
        return x

class ConvSelfAttention(nn.Module):
    def __init__(self, dim=256, head=2, qkv_bias=False, patch_size=16):
        super().__init__()
        # (b*t, 256, H/4, W/4)
        # self.conv_qkv = nn.Sequential(
        #     nn.AvgPool2d(2),
        #     nn.Conv2d(dim, dim*2*head, 3, stride=1, padding=1, bias=qkv_bias),
        #     nn.AvgPool2d(2),
        # )
        self.conv_qkv = nn.Conv2d(dim, dim*2*head, patch_size, patch_size, patch_size//2, bias=qkv_bias)
        # (b*t, qkv, H', W')
        self.conv_v = nn.Conv2d(dim, dim*head, 1, bias=qkv_bias)
        self.head = head
        self.ch_qkv = dim
        self.merge_v = nn.Conv2d(dim*self.head, dim, 1)
        self.qk_scale = dim ** -0.5
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size, padding=patch_size//2)
    
    def forward(self, x):
        # (b, t, c, h, w)
        # print(x.shape)
        b, t, c, h, w = x.shape
        qkv = rearrange(x, 'b t c h w -> (b t) c h w')
        # v = (rearrange(x, 'b t c h w -> (b t) c h w'))
        v = self.conv_v(qkv)
        # v = torch.tile(qkv, (1, self.head, 1, 1))
        # print(v.shape)
        v = self.unfold(v) # ((b t) P*P*c*m h'*w')

        v = rearrange(v, '(b t) (m c) w -> b m (t w) c', b=b, t=t, m=self.head) # t*patch attention
        # v = rearrange(v, '(b t) (m c) w -> b m w t c', b=b, t=t, m=self.head) # patchwise t attention

        # print(qkv.shape)
        
        qkv = self.conv_qkv(qkv) # (b*t, qkv, H', W')
        qkv = rearrange(qkv, "(b t) (q m c) h w -> q b m (t h w) c", b=b, t=t, q=2, m=self.head, c=self.ch_qkv)
        # patchwise t attention
        # qkv = rearrange(qkv, "(b t) (q m c) h w -> q b m (h w) t c", b=b, t=t, q=2, m=self.head, c=self.ch_qkv)
        q, k = qkv[0], qkv[1]
        A = q @ k.transpose(-2, -1)
        A = A.softmax(dim=-1)
        out = A @ v

        # out = rearrange(out, 'b m t (c h w) ->(b t) (m c) h w', b=b, t=t, c=c, h=h, w=w, m=self.head)

        out = rearrange(out, 'b m (t w) c ->(b t m) c w', t=t)
        # out = rearrange(out, 'b m w t c ->(b t m) c w',)
        out = F.fold(out, (h, w), kernel_size=self.patch_size, stride=self.patch_size, padding=self.patch_size//2)
        out = rearrange(out, '(b m) c h w -> b (c m) h w', m=self.head)
        out = self.merge_v(out)
        out = rearrange(out, '(b t) c h w -> b t c h w', b=b, t=t, c=c, h=h, w=w)

        # print(out.shape)
        # raise
        return out

class ST_TransformerBlk(nn.Module):
    def __init__(self, ch_in=256, ch_out=None, up_scale=1, seq_size=7, norm=nn.LayerNorm):
        super().__init__()
        if ch_out is None:
            ch_out = ch_in
        self.ch_out = ch_out
        self.up_scale = up_scale

        self.act = nn.GELU()
        self.sa = ConvSelfAttention(ch_in)
        self.feedforward = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, 1, 1),
            self.act
        )
        self.skip = nn.Conv2d(ch_in, ch_out, 1, 1) if ch_in != ch_out else nn.Identity()
        self.up = nn.UpsamplingBilinear2d(scale_factor=up_scale) if up_scale > 1 else nn.Identity()
        self.pe = nn.Parameter(torch.zeros(1, seq_size, ch_in, 1, 1))
        p = 0.
        self.drop = nn.Dropout(p)
        # norm = nn.Identity
        if norm == nn.LayerNorm:
            self.init_flg = True
        else:
            self.init_flg = False
            norm = norm()
        self.norm1 = norm
        self.norm2 = norm
        
    def forward(self, x):
        if self.init_flg:
            self.init_flg = False
            c, h, w = x.shape[-3:]
            self.norm1 = self.norm1((c, h, w)).cuda()
            self.norm2 = self.norm2((self.ch_out, h*self.up_scale, w*self.up_scale)).cuda()
        
        x = x + self.pe
        # out = x + self.sa(x)
        out = x + self.drop(self.norm1(self.sa(x)))
        # out = F.layer_norm(out, out.shape[-3:])

        b = out.size(0)
        out = rearrange(out, 'b t c h w -> (b t) c h w')
        out = self.up(out)
        out = self.skip(out) + self.drop(self.norm2(self.feedforward(out)))
        # out = F.layer_norm(out, out.shape[-3:])
        
        out = rearrange(out, '(b t) c h w -> b t c h w', b=b)
        return out

class To4D(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return rearrange(x, 'b t c h w -> (b t) c h w')

class To5D(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x):
        return rearrange(x, '(b t) c h w -> b t c h w', t=self.t)


class DeepLabV3Plus_SA2(nn.Module):
    def __init__(self, num_classes,
        hidden_channels: Tuple[int, int] = (3, 8),
        continue_num = 8, backbone = "resnet34"
        ):
        # TODO: remove args (& hidden channels)
        super().__init__()
        aspp_channel = 128
        self.unet1 = smp.DeepLabV3Plus(
            encoder_name=backbone,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            decoder_channels=aspp_channel,
            classes=1, # model output channels (number of classes in your dataset),
        )
        self.unet1.forward = MethodType(self.deeplab_forward, self.unet1)

        def get_conv4x_layers(in_ch, out_ch, kernel_size=3):
            half_kernel = kernel_size//2
            return nn.Sequential(
                nn.Conv2d(hidden_channels[-1], hidden_channels[-1]//2, kernel_size=kernel_size, padding=half_kernel, stride=half_kernel),
                nn.ReLU(),
                nn.UpsamplingBilinear2d(scale_factor=2),
                
                nn.Conv2d(hidden_channels[-1]//2, out_channels=out_ch, kernel_size=kernel_size, padding=half_kernel, stride=half_kernel),
                nn.UpsamplingBilinear2d(scale_factor=2),
            )

        self.len = continue_num
        
        self.act = nn.GELU()
        self.transformer_blks = nn.Sequential(
            ST_TransformerBlk(aspp_channel, aspp_channel, up_scale=1),
            To4D(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(aspp_channel, aspp_channel//2, 3, 1, 1),
            self.act,
            To5D(continue_num),

            ST_TransformerBlk(aspp_channel//2, aspp_channel//2, up_scale=1),
            To4D(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(aspp_channel//2, aspp_channel//4, 3, 1, 1),
            self.act,
            To5D(continue_num),
        )
        
        # self.transformer_blks = nn.Sequential(
        #     ST_TransformerBlk(aspp_channel, aspp_channel//2, up_scale=2),
        #     ST_TransformerBlk(aspp_channel//2, aspp_channel//4, up_scale=2),
        # )

        self.upconv = nn.Conv2d(aspp_channel//4, 1, 3, 1, 1)
        # self.finalconv = nn.Conv2d(continue_num, 1, 1,)
    
    @staticmethod
    def deeplab_forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features) # (b, 256, 88, 104)
        return decoder_output

    def forward(self, input, other_frame):
        # print(input.shape)
        temporal_mask = torch.tensor([]).cuda()
        b = other_frame.size(0)
        other_frame = rearrange(other_frame, 'b t c h w -> (b t) c h w')
        continue_list = self.unet1(other_frame)
        continue_list = rearrange(continue_list, '(b t) c h w -> b t c h w', b=b)
        
        out = self.transformer_blks(continue_list)
        out = rearrange(out, 'b t c h w -> (b t) c h w')
        out = rearrange(self.upconv(out), '(b t) c h w -> b t c h w', b=b)
        final_predict, temporal_mask = out[:, out.size(1)//2], out.squeeze(2)

        # final_out = self.finalconv(rearrange(out, 'b t c h w -> b (t c) h w'))
        # final_predict, temporal_mask = final_out, out.squeeze(2)

        return temporal_mask, final_predict

