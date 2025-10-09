# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F

from resnet_adaibn import resnet50_adaibn_a, pretrained_in_weight

# -------------------- init & small helpers --------------------
def weights_init_kaiming(m):
    name = m.__class__.__name__
    if 'Conv' in name:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif 'Linear' in name:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif 'BatchNorm1d' in name:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    name = m.__class__.__name__
    if 'Linear' in name:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    elif 'BatchNorm1d' in name:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def _as_str(x, default='normal'):
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return str(x[-1])
    if isinstance(x, str):
        return x
    return default

def assign_adain_params(adain_params_w, adain_params_b, model, dim=32):
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params_b[:, :dim].contiguous()
            std  = adain_params_w[:, :dim].contiguous()
            m.bias = mean.view(-1)
            m.weight = std.view(-1)
            if adain_params_w.size(1) > dim:
                adain_params_b = adain_params_b[:, dim:]
                adain_params_w = adain_params_w[:, dim:]

def spade_norm(layer, x, mod_f):
    _, c, _, _ = x.shape
    half = int(0.5 * c)
    x_in, x_bn = torch.split(x, [half, c - half], dim=1)
    out1 = layer.IN(x_in.contiguous())
    out1 = out1 * (mod_f[0] + 1) + mod_f[1]
    out2 = layer.BN(x_bn.contiguous())
    return torch.cat((out1, out2), 1)

def extract_spade_feature(layer, x, mod_f):
    feats = [x]
    for i, block in enumerate(layer.children()):
        residual = feats[-1]
        out = block.conv1(feats[-1])
        if hasattr(block.bn1, 'IN') and mod_f[i][0] is not None:
            out = spade_norm(block.bn1, out, mod_f[i])
        else:
            out = block.bn1(out)
        out = block.relu(out)

        out = block.conv2(out); out = block.bn2(out); out = block.relu(out)
        out = block.conv3(out); out = block.bn3(out)

        if block.downsample is not None:
            residual = block.downsample(residual)
        out = block.relu(out + residual)
        feats.append(out)
    return feats[-1]

# -------------------- optional text encoder --------------------
from transformers import BertTokenizer, BertModel

def _resolve_text_model_path(user_path: str | None) -> str:
    if user_path:
        return user_path
    env = os.getenv("TEXT_MODEL_NAME_OR_PATH")
    return env.strip() if env else "bert-base-uncased"

class SimpleTextEncoder(nn.Module):
    def __init__(self, model_name_or_path: str | None = None, output_dim: int = 512):
        super().__init__()
        path = _resolve_text_model_path(model_name_or_path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.bert = BertModel.from_pretrained(path)
        self.proj = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, text_list):
        device = next(self.bert.parameters()).device
        tok = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
        out = self.bert(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)

# -------------------- building blocks --------------------
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, bottleneck=512, return_f=False):
        super().__init__()
        self.return_f = return_f
        self.Linear = nn.Linear(input_dim, bottleneck)
        self.bnorm  = nn.BatchNorm1d(bottleneck)
        self.dropout = nn.Dropout(p=droprate)
        init.kaiming_normal_(self.Linear.weight.data, a=0, mode='fan_out')
        init.constant_(self.Linear.bias.data, 0.0)
        init.normal_(self.bnorm.weight.data, 1.0, 0.02)
        init.constant_(self.bnorm.bias.data, 0.0)
        self.classifier = nn.Sequential(nn.Linear(bottleneck, class_num))
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.Linear(x)
        x = self.bnorm(x)
        x = self.dropout(x)
        if self.return_f:
            f = x
            return [self.classifier(x), f]
        return self.classifier(x)

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu', init_mode='kaiming'):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        mode = _as_str(init_mode, 'kaiming')
        if mode == 'kaiming':
            init.kaiming_normal_(self.fc.weight.data, a=0, mode='fan_out'); init.constant_(self.fc.bias.data, 0.0)
        elif mode == 'normal':
            init.normal_(self.fc.weight.data, std=0.001); init.constant_(self.fc.bias.data, 0.0)
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(output_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = None
        acts = {
            'relu': nn.ReLU(inplace=True), 'lrelu': nn.LeakyReLU(0.2, inplace=True),
            'prelu': nn.PReLU(), 'selu': nn.SELU(inplace=True), 'tanh': nn.Tanh(), 'none': None
        }
        self.activation = acts.get(activation, nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.fc(x)
        if self.norm is not None:
            out = self.norm(out).view(out.size(0), out.size(1))
        if self.activation is not None:
            out = self.activation(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='in', activ='relu'):
        super().__init__()
        layers = [LinearBlock(input_dim, dim, norm=norm, activation=activ, init_mode='kaiming')]
        for _ in range(max(n_blk - 2, 0)):
            layers += [LinearBlock(dim, dim, norm=norm, activation=activ, init_mode='kaiming')]
        self.model = nn.Sequential(*layers)
        self.Gen = nn.Sequential(LinearBlock(dim, output_dim, norm='none', activation='none', init_mode='normal'))

    def forward(self, x):
        x = self.model(x.view(x.size(0), -1))
        return self.Gen(x)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, norm='none', activation='relu', init_mode='normal'):
        super().__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, k, s, p, bias=True)
        mode = _as_str(init_mode, 'normal')
        if mode == 'kaiming':
            init.kaiming_normal_(self.conv2d.weight.data, mode='fan_in', nonlinearity="relu"); init.constant_(self.conv2d.bias.data, 0.0)
        elif mode == 'normal':
            init.normal_(self.conv2d.weight.data, std=0.001); init.constant_(self.conv2d.bias.data, 0.0)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_c)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_c)
        elif norm == 'ln':
            self.norm = nn.LayerNorm([out_c, 64, 64])
        else:
            self.norm = None
        acts = {
            'relu': nn.ReLU(inplace=True), 'lrelu': nn.LeakyReLU(0.2, inplace=True),
            'prelu': nn.PReLU(), 'selu': nn.SELU(inplace=True),
            'tanh': nn.Tanh(), 'none': None
        }
        self.activation = acts.get(activation, nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv2d(x)
        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out

class MOD_(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1,
                 dim=0, n_blk=0, norm='in', activ='relu', init_mode='normal'):
        super().__init__()
        self.Gen = nn.Sequential(
            ConvBlock(input_dim, output_dim, kernel_size, stride, padding, norm=norm, activation='none', init_mode=init_mode)
        )
    def forward(self, x):
        return self.Gen(x)

# -------------------- Pt_ResNet50 (contrastive / weather) --------------------
class Pt_ResNet50(nn.Module):
    def __init__(
        self,
        pool='avg',
        init_model=None,
        norm='ada-ibn',
        init_mode='normal',
        btnk=[1, 0, 1],
        conv_norm='in',
        weather_num_classes: int = 11,
        text_model_name_or_path: str | None = None,
    ):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.layer2[3].relu = nn.Sequential()
        if pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.model = model_ft
        self.pool = pool

        self.norm = norm
        self.btnk = btnk

        if norm == 'spade':
            if btnk[0] == 1:
                self.layer1_0_w = MOD_(512, 32, norm=conv_norm, init_mode=init_mode)
                self.layer1_0_b = MOD_(512, 32, norm=conv_norm, init_mode=init_mode)
            if btnk[1] == 1:
                self.layer1_1_w = MOD_(512, 32, norm=conv_norm, init_mode=init_mode)
                self.layer1_1_b = MOD_(512, 32, norm=conv_norm, init_mode=init_mode)
            if btnk[2] == 1:
                self.layer1_2_w = MOD_(512, 32, norm=conv_norm, init_mode=init_mode)
                self.layer1_2_b = MOD_(512, 32, norm=conv_norm, init_mode=init_mode)
            if len(btnk) > 3 and btnk[3] == 1:
                self.layer2_0_w = MOD_(512, 64, norm=conv_norm, init_mode=init_mode)
                self.layer2_0_b = MOD_(512, 64, norm=conv_norm, init_mode=init_mode)
            if len(btnk) > 4 and btnk[4] == 1:
                self.layer2_1_w = MOD_(512, 64, norm=conv_norm, init_mode=init_mode)
                self.layer2_1_b = MOD_(512, 64, norm=conv_norm, init_mode=init_mode)
            if len(btnk) > 5 and btnk[5] == 1:
                self.layer2_2_w = MOD_(512, 64, norm=conv_norm, init_mode=init_mode)
                self.layer2_2_b = MOD_(512, 64, norm=conv_norm, init_mode=init_mode)
            if len(btnk) > 6 and btnk[6] == 1:
                self.layer2_3_w = MOD_(512, 64, norm=conv_norm, init_mode=init_mode)
                self.layer2_3_b = MOD_(512, 64, norm=conv_norm, init_mode=init_mode)
        else:
            self.layer1_w   = MLP(512, 32, 512, 3, norm='none', activ='lrelu')
            self.layer1_b   = MLP(512, 32, 512, 3, norm='none', activ='lrelu')
            self.layer1_2_w = MLP(512, 32, 512, 3, norm='none', activ='lrelu')
            self.layer1_2_b = MLP(512, 32, 512, 3, norm='none', activ='lrelu')

        # text & projections
        self.text_encoder = SimpleTextEncoder(model_name_or_path=text_model_name_or_path, output_dim=512)
        self.img_projection = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512))
        self.txt_projection = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512))

        # weather classifier
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Dropout(p=0.5), nn.Linear(512, weather_num_classes))
        self.classifier.apply(weights_init_classifier)

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def _pool_feat(self, x2):
        x3 = self.model.avgpool2(x2)
        return x3.flatten(1)

    def forward(self, x, captions=None):
        x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x)
        x0 = self.model.maxpool(x)
        x1 = self.model.layer1(x0)
        x2 = self.model.layer2(x1)

        img_feat = self._pool_feat(x2)   # [B, 512] after flatten
        img_emb  = self.img_projection(img_feat)
        txt_emb  = None
        if captions is not None:
            txt_feat = self.text_encoder(captions)
            txt_emb  = self.txt_projection(txt_feat)

        if self.norm == 'spade':
            B, C, H, W = x2.size()
            x2_ = F.interpolate(x2, size=(2 * H, 2 * W), mode='nearest')
            w1 = b1 = w1_1 = b1_1 = w1_2 = b1_2 = None
            w2_0 = b2_0 = w2_1 = b2_1 = w2_2 = b2_2 = w2_3 = b2_3 = None

            if hasattr(self, 'layer1_0_w'):
                w1 = self.layer1_0_w(x2_); b1 = self.layer1_0_b(x2_)
            if hasattr(self, 'layer1_1_w'):
                w1_1 = self.layer1_1_w(x2_); b1_1 = self.layer1_1_b(x2_)
            if hasattr(self, 'layer1_2_w'):
                w1_2 = self.layer1_2_w(x2_); b1_2 = self.layer1_2_b(x2_)
            if hasattr(self, 'layer2_0_w'):
                w2_0 = self.layer2_0_w(x2_); b2_0 = self.layer2_0_b(x2_)
            if hasattr(self, 'layer2_1_w'):
                w2_1 = self.layer2_1_w(x2);  b2_1 = self.layer2_1_b(x2)
            if hasattr(self, 'layer2_2_w'):
                w2_2 = self.layer2_2_w(x2);  b2_2 = self.layer2_2_b(x2)
            if hasattr(self, 'layer2_3_w'):
                w2_3 = self.layer2_3_w(x2);  b2_3 = self.layer2_3_b(x2)

            out_w = self.classifier(img_feat)
            mod_f = [[w1, b1], [w1_1, b1_1], [w1_2, b1_2]], [[w2_0, b2_0], [w2_1, b2_1], [w2_2, b2_2], [w2_3, b2_3]]
            return mod_f, out_w, img_emb, txt_emb


        w1  = self.layer1_w(img_feat);   b1  = self.layer1_b(img_feat)
        w1_2 = self.layer1_2_w(img_feat); b1_2 = self.layer1_2_b(img_feat)
        w2 = b2 = w3 = b3 = w1_1 = b1_1 = w2_3 = b2_3 = w3_5 = b3_5 = w3_3 = b3_3 = None
        out_w = self.classifier(img_feat)
        return [w1, w1_1, w1_2], [b1, b1_1, b1_2], [w2, w2_3], [b2, b2_3], [w3, w3_3, w3_5], [b3, b3_3, b3_5], out_w, img_emb, txt_emb

# -------------------- backbones --------------------
class ft_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', norm='bn', adain='a'):
        super().__init__()
        if norm == 'bn':
            model_ft = models.resnet50(pretrained=True)
        elif norm == 'ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        elif norm == 'ada-ibn':
            model_ft = resnet50_adaibn_a(pretrained=True, adain=adain)
        else:
            model_ft = models.resnet50(pretrained=True)

        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

        self.model = model_ft
        self.pool = pool
        self.norm = norm

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x):
        x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x); x = self.model.layer2(x); x = self.model.layer3(x); x = self.model.layer4(x)
        if self.pool == 'avg+max':
            a = self.model.avgpool2(x); m = self.model.maxpool2(x)
            x = torch.cat((a, m), dim=1).view(x.size(0), -1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x).view(x.size(0), -1)
        elif self.pool == 'max':
            x = self.model.maxpool2(x).view(x.size(0), -1)
        return x

class ft_net_spade(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', norm='bn', adain='a'):
        super().__init__()
        if norm in ('ada-ibn', 'spade'):
            model_ft = resnet50_adaibn_a(pretrained=True, adain=adain)
        elif norm == 'ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        else:
            model_ft = models.resnet50(pretrained=True)

        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        if pool == 'avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

        self.model = model_ft
        self.pool = pool
        self.norm = norm

        if init_model is not None:
            self.model = init_model.model
            self.pool = init_model.pool

    def forward(self, x, mod_f):
        x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x); x = self.model.maxpool(x)
        x = extract_spade_feature(self.model.layer1, x, mod_f[0]) if self.norm == 'spade' else self.model.layer1(x)
        x = extract_spade_feature(self.model.layer2, x, mod_f[1]) if self.norm == 'spade' else self.model.layer2(x)
        x = self.model.layer3(x); x = self.model.layer4(x)
        if self.pool == 'avg+max':
            a = self.model.avgpool2(x); m = self.model.maxpool2(x)
            x = torch.cat((a, m), dim=1).view(x.size(0), -1)
        elif self.pool == 'avg':
            x = self.model.avgpool2(x).view(x.size(0), -1)
        elif self.pool == 'max':
            x = self.model.maxpool2(x).view(x.size(0), -1)
        return x


class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=1, pool='avg',
                 share_weight=False, norm='bn', adain='a', btnk=[1, 0, 1]):
        super().__init__()
        self.norm = norm
        self.adain = adain

        self.model_1 = ft_net_spade(class_num, stride=stride, pool=pool, norm=norm, adain=adain) if norm == 'spade' \
                       else ft_net(class_num, stride=stride, pool=pool, norm=norm, adain=adain)
        self.model_2 = self.model_1 if share_weight else ft_net(class_num, stride=stride, pool=pool)

        self.classifier = ClassBlock(2048, class_num, droprate)
        if pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)

        if norm in ('ada-ibn', 'spade'):
            self.pt_model = Pt_ResNet50(norm=norm, init_mode='normal', btnk=btnk)
            self._w1, self._b1, self._w2, self._b2, self._w3, self._b3 = pretrained_in_weight(True)

    def forward(self, x1, x2):
        # view1
        if x1 is None:
            y1 = None
        else:
            if self.norm == 'ada-ibn':
                sw1, sb1, sw2, sb2, sw3, sb3, sout_w, *_ = self.pt_model(x1)
                if self.adain == 'a':
                    assign_adain_params(sw1[0] + self._w1[0], sb1[0] + self._b1[0], self.model_1.model.layer1[0], 32)
                    assign_adain_params(sw1[2] + self._w1[2], sb1[2] + self._b1[2], self.model_1.model.layer1[2], 32)
                x1 = self.model_1(x1)
            elif self.norm == 'spade':
                smod_f, sout_w, *_ = self.pt_model(x1)
                x1 = self.model_1(x1, smod_f)
            else:
                x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        # view2
        if x2 is None:
            y2 = None
        else:
            if self.norm == 'ada-ibn':
                gw1, gb1, gw2, gb2, gw3, gb3, gout_w, *_ = self.pt_model(x2)
                if self.adain == 'a':
                    assign_adain_params(gw1[0] + self._w1[0], gb1[0] + self._b1[0], self.model_2.model.layer1[0], 32)
                    assign_adain_params(gw1[2] + self._w1[2], gb1[2] + self._b1[2], self.model_2.model.layer1[2], 32)
                x2 = self.model_2(x2)
            elif self.norm == 'spade':
                gmod_f, gout_w, *_ = self.pt_model(x2)
                x2 = self.model_2(x2, gmod_f)
            else:
                x2 = self.model_2(x2)
            y2 = self.classifier(x2)

        if self.norm in ('ada-ibn', 'spade'):
            if not self.training:
                return y1, y2
            return y1, y2, sout_w, gout_w
        return y1, y2

class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride=2, pool='avg', share_weight=False,
                 norm='bn', adain='a', circle=False, btnk=[1, 0, 1], conv_norm='none'):
        super().__init__()
        self.norm = norm
        self.adain = adain
        self.circle = circle

        self.model_1 = ft_net_spade(class_num, stride=stride, pool=pool, norm=norm, adain=adain) if norm == 'spade' \
                       else ft_net(class_num, stride=stride, pool=pool, norm=norm, adain=adain)
        self.model_2 = ft_net(class_num, stride=stride, pool=pool)
        self.model_3 = self.model_1 if share_weight else ft_net(class_num, stride=stride, pool=pool)

        if norm in ('ada-ibn', 'spade'):
            self.pt_model = Pt_ResNet50(norm=norm, init_mode='normal', btnk=btnk, conv_norm=conv_norm)
        if norm == 'ada-ibn':
            self._w1, self._b1, self._w2, self._b2, self._w3, self._b3 = pretrained_in_weight(True)

        self.classifier = ClassBlock(2048, class_num, droprate, return_f=circle)
        if pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)

    def forward(self, x1, x2, x3, x4=None, captions=None):
        # view1
        if x1 is None:
            y1 = None
        else:
            if self.norm == 'ada-ibn':
                sw1, sb1, sw2, sb2, sw3, sb3, sout_w, *_ = self.pt_model(x1)
                if self.adain == 'a':
                    assign_adain_params(sw1[0] + self._w1[0], sb1[0] + self._b1[0], self.model_1.model.layer1[0], 32)
                    assign_adain_params(sw1[2] + self._w1[2], sb1[2] + self._b1[2], self.model_1.model.layer1[2], 32)
                x1 = self.model_1(x1)
            elif self.norm == 'spade':
                smod_f, sout_w, *_ = self.pt_model(x1)
                x1 = self.model_1(x1, smod_f)
            else:
                x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        # view2
        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)

        # view3
        if x3 is None:
            y3 = None
        else:
            if self.norm == 'ada-ibn':
                dw1, db1, dw2, db2, dw3, db3, dout_w, *_ = self.pt_model(x3)
                if self.adain == 'a':
                    assign_adain_params(dw1[0] + self._w1[0], db1[0] + self._b1[0], self.model_3.model.layer1[0], 32)
                    assign_adain_params(dw1[2] + self._w1[2], db1[2] + self._b1[2], self.model_3.model.layer1[2], 32)
                x3 = self.model_3(x3)
            elif self.norm == 'spade':
                dmod_f, dout_w, *_ = self.pt_model(x3, captions)
                x3 = self.model_3(x3, dmod_f)
            else:
                x3 = self.model_3(x3)
            y3 = self.classifier(x3)

        if x4 is None:
            if self.norm in ('ada-ibn', 'spade'):
                return y1, y2, y3
            return y1, y2, y3
        else:
            x4 = self.model_2(x4)
            y4 = self.classifier(x4)
            if self.norm in ('ada-ibn', 'spade'):
                return y1, y2, y3, y4, sout_w, dout_w
            return y1, y2, y3, y4
