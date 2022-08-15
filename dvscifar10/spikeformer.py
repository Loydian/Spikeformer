import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from utils import DropPath, trunc_normal_
from einops import rearrange
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from neuron import MultiStepParametricLIAFNode
from spikingjelly.clock_driven import layer


def conv(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False, bn=True,
         act_layer=MultiStepParametricLIFNode):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
        ),
        act_layer(detach_reset=True)
    )


class SEWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), kernel_size=(3, 3), padding=(1, 1), conv_bias=False,
                 act_layer=MultiStepParametricLIFNode):
        super().__init__()

        self.residual_function = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                 bias=conv_bias, act_layer=act_layer),
            conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                 bias=conv_bias, act_layer=act_layer)
        )

        self.shortcut = nn.Sequential()

        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer.SeqToANNContainer(
                    nn.AvgPool2d(kernel_size=stride, stride=stride)),
                conv(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=conv_bias,
                     act_layer=act_layer)
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=MultiStepParametricLIFNode, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Linear(in_features, hidden_features)
            ),
            act_layer(detach_reset=True),
            layer.SeqToANNContainer(
                nn.Linear(hidden_features, out_features),
                nn.Dropout(drop)
            )
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=MultiStepParametricLIFNode,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)

        # Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W):
        num_spatial_tokens = x.size(1) // T
        H = num_spatial_tokens // W

        # Temporal
        xt = rearrange(x, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
        res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
        res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x + res_temporal

        # Spatial
        xs = xt
        xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
        res_spatial = self.drop_path(self.attn(self.norm1(xs)))

        res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
        res = res_spatial
        x = xt

        # Mlp
        x = x + res
        res = rearrange(x, 'b (h w t) m -> t b (h w) m', b=B, h=H, w=W, t=T)
        res = self.drop_path(self.mlp(self.norm2(res)))
        res = rearrange(res, ' t b (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
        x = x + res
        return x


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 act_layer=MultiStepParametricLIFNode,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        if isinstance(stride, int):
            n_filter_list = [n_input_channels] + \
                            [n_output_channels // 2 ** (n_conv_layers - i - 1) for i in range(n_conv_layers - 1)] + \
                            [n_output_channels]
        else:
            temp = [n_output_channels // 2 ** ((n_conv_layers // 2) - i - 1) for i in
                    range(0, (n_conv_layers // 2) - 1)]
            n_filter_list = [n_input_channels]
            for i in temp:
                n_filter_list += [i]
                n_filter_list += [i]
            n_filter_list += [n_output_channels, n_output_channels]

        block = SEWBlock
        index = 0
        self.conv_layers = nn.Sequential(
            *[
                block(n_filter_list[i], n_filter_list[i + 1], stride=(stride, stride)
                      if isinstance(stride, int) else (stride[i - index], stride[i - index]), act_layer=act_layer,
                      kernel_size=kernel_size, padding=(padding, padding), conv_bias=conv_bias)
                for i in range(index, n_conv_layers)
            ]
        )

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=2, height=224, width=224):
        with torch.no_grad():
            return self.forward(torch.zeros((1, 1, n_channels, height, width)))[0].shape[1]

    def forward(self, x):
        B, T, C, H, W = x.shape
        # x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = rearrange(x, 'b t c h w -> t b c h w')
        x = self.conv_layers(x)
        x = rearrange(x, 't b c h w -> (b t) c h w')
        W = x.size(-1)
        x = self.flattener(x).transpose(-2, -1)
        return x, T, W

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class spikeformer(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1,
                 n_conv_layers=1, n_input_channels=3, num_classes=1000, embed_dim=768, depth=12, img_size=224,
                 num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, num_frames=8,
                 act_layer=MultiStepParametricLIFNode):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_layer=act_layer,
            n_conv_layers=n_conv_layers,
            conv_bias=False
        )
        num_patches = self.patch_embed.sequence_length(n_channels=n_input_channels,
                                                       height=img_size,
                                                       width=img_size)

        self.num_patches = num_patches

        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        # Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        self.attention_pool = nn.Linear(self.embed_dim, 1)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        # initialization of temporal attention weights
        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if 'Block' in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed'}

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)

        # resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            other_pos_embed = pos_embed[0, :, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Time Embeddings
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        # Resizing time embeddings in case they don't match
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=T, mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        # Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        x = self.norm(x)

        x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Spikeformer(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1,
                 n_conv_layers=1, n_input_channels=3, img_size=224,
                 drop_rate=0.,
                 num_classes=1000, num_frames=8,
                 mlp_ratio=4,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 embed_dim=768, depth=12, num_heads=12,
                 act_layer=MultiStepParametricLIFNode):
        super(Spikeformer, self).__init__()
        self.model = spikeformer(
            kernel_size=kernel_size, stride=stride, padding=padding, n_conv_layers=n_conv_layers,
            n_input_channels=n_input_channels, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
            img_size=img_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=act_layer,
            num_frames=num_frames)

        self.num_patches = self.model.num_patches

    def forward(self, x):
        x = self.model(x)
        return x


# DVS-CIFAR10
def PLIF_spikeformer_7_3x2x3(n_input_channels=2, num_classes=10, img_size=128, num_frame=4):
    return Spikeformer(kernel_size=3, n_conv_layers=6, img_size=img_size, num_classes=num_classes, depth=7, num_heads=4,
                       mlp_ratio=2,
                       embed_dim=256, stride=[2, 1, 2, 1, 2, 1], padding=1,
                       n_input_channels=n_input_channels, num_frames=num_frame)


# DVS-Gesture
def PLIF_spikeformer_7_5x3x1(n_input_channels=2, num_classes=11, img_size=128, num_frame=16):
    return Spikeformer(kernel_size=5, n_conv_layers=3, img_size=img_size, num_classes=num_classes, depth=7, num_heads=4,
                       mlp_ratio=2,
                       embed_dim=256, stride=2, padding=2,
                       n_input_channels=n_input_channels, num_frames=num_frame)


# ImageNet
def PLIAF_spikeformer_7_3x2x4(n_input_channels=3, num_classes=1000, img_size=224, num_frame=4):
    return Spikeformer(kernel_size=3, n_conv_layers=8, img_size=img_size, num_classes=num_classes, depth=7, num_heads=8,
                       mlp_ratio=3, act_layer=MultiStepParametricLIAFNode,
                       embed_dim=512, stride=[2, 1, 2, 1, 2, 1, 2, 1], padding=1,
                       n_input_channels=n_input_channels, num_frames=num_frame)
