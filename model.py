import argparse
import math
import torch
import os
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
from resnet_adaibn import resnet50_adaibn_a, pretrained_in_weight
from transformers import BertTokenizer, BertModel
from transformers import BertForMaskedLM
from torch.nn.functional import cosine_similarity
import sys

# from resnet_paibn import resnet50_ibn_a_adapter
######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def assign_adain_params(adain_params_w, adain_params_b, model, dim=32, init_w=None, init_b=None):
    # assign the adain_params to the AdaIN layers in model
    # dim = self.output_dim
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params_b[:, :dim].contiguous()
            std = adain_params_w[:, :dim].contiguous()
            m.bias = mean.view(-1)
            m.weight = std.view(-1)
            if adain_params_w.size(1) > dim:  # Pop the parameters
                adain_params_b = adain_params_b[:, dim:]
                adain_params_w = adain_params_w[:, dim:]

def spade_norm(layer, x, mod_f):
    _, channel, _, _ = x.shape
    half = int(0.5 * channel)
    split = torch.split(x, half, 1)
    # if mod_f[0].size()[2:] != split[0].size()[2:]:
    #     mod_f[0] = F.interpolate(mod_f[0], size=split[0].size()[2:], mode='nearest')
    #     mod_f[1] = F.interpolate(mod_f[1], size=split[0].size()[2:], mode='nearest')
    out1 = layer.IN(split[0].contiguous())
    out1 = out1 * (mod_f[0] + 1) + mod_f[1]
    # out1 = out1 * (mod_f[0] + 1)
   # out1 = out1 * torch.exp(mod_f[0]) + mod_f[1]
    out2 = layer.BN(split[1].contiguous())
    out = torch.cat((out1, out2), 1)
    return out

def extract_spade_feature(layer, x, mod_f):
    input = [x]
    for i, c_layer in enumerate(layer.children()):
        residual = input[-1]
        out = c_layer.conv1(input[-1])
        if hasattr(c_layer.bn1, 'IN') and mod_f[i][0] != None:
            out = spade_norm(c_layer.bn1, out, mod_f[i])
        else:
            out = c_layer.bn1(out)
        out = c_layer.relu(out)

        out = c_layer.conv2(out)
        out = c_layer.bn2(out)
        out = c_layer.relu(out)

        out = c_layer.conv3(out)
        out = c_layer.bn3(out)

        if c_layer.downsample is not None:
            residual = c_layer.downsample(residual)
        out += residual

        out = c_layer.relu(out)
        input.append(out)
    return input[-1]




class SimpleTextEncoder(nn.Module):
    def __init__(self, model_name="/XXX/bert-base-uncased", output_dim=512):
        super(SimpleTextEncoder, self).__init__()
        from transformers import BertTokenizer, BertModel
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, output_dim)
        
    def forward(self, text_list):
        device = next(self.bert.parameters()).device
        encoded = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        projected = self.proj(cls_embeddings)               # [B, output_dim]
        return projected

    


class GammaBetaMLP(nn.Module):
    def __init__(self, input_dim=512, channel_dim=512):
        super(GammaBetaMLP, self).__init__()
        self.gamma_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, channel_dim)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, channel_dim)
        )

    def forward(self, text_features):
        gamma = self.gamma_mlp(text_features)
        beta = self.beta_mlp(text_features)
        return gamma, beta

def adain(x, gamma, beta, eps=1e-5):
    B, C, H, W = x.size()
    gamma = gamma.view(B, C, 1, 1)
    beta = beta.view(B, C, 1, 1)
    mean = x.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    std = x.view(B, C, -1).std(dim=2).view(B, C, 1, 1) + eps
    return gamma * ((x - mean) / std) + beta




# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, num_bottleneck=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.Linear = nn.Linear(input_dim, num_bottleneck)
        self.bnorm = nn.BatchNorm1d(num_bottleneck)
        self.dropout = nn.Dropout(p=droprate)
        init.kaiming_normal_(self.Linear.weight.data, a=0, mode='fan_out')
        init.constant_(self.Linear.bias.data, 0.0)
        init.normal_(self.bnorm.weight.data, 1.0, 0.02)
        init.constant_(self.bnorm.bias.data, 0.0)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.Linear(x)
        x = self.bnorm(x)
        x = self.dropout(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu', init_mode='kaiming'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        if init_mode == 'kaiming':
            init.kaiming_normal_(self.fc.weight.data, a=0, mode='fan_out')
            init.constant_(self.fc.bias.data, 0.0)
        elif init_mode == 'normal':
            init.normal_(self.fc.weight.data, std=0.001)
            init.constant_(self.fc.bias.data, 0.0)
        elif init_mode == 'none':
            pass
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            #reshape input
            # out = out.unsqueeze(1)
            out = self.norm(out)
            out = out.view(out.size(0),out.size(1))
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='in', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ, init_mode='kaiming')]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ, init_mode='kaiming')]
        self.model = nn.Sequential(*self.model)

        self.Gen = []
        self.Gen += [LinearBlock(dim, output_dim, norm='none', activation='none', init_mode='normal')]
        self.Gen = nn.Sequential(*self.Gen)

        # self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        # self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model(x.view(x.size(0), -1))
        x = self.Gen(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, norm='none', activation='relu', init_mode='normal'):
        super(ConvBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.conv2d = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=use_bias)
        if init_mode == 'kaiming':
            init.kaiming_normal_(self.conv2d.weight.data, mode='fan_in', nonlinearity="relu")
            init.constant_(self.conv2d.bias.data, 0.0)
        elif init_mode == 'normal':
            init.normal_(self.conv2d.weight.data, std=0.001)
            init.constant_(self.conv2d.bias.data, 0.0)
        elif init_mode == 'none':
            pass
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm([norm_dim,64,64])
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.conv2d(x)
        if self.norm:
            #reshape input
            # out = out.unsqueeze(1)
            out = self.norm(out)
            # out = out.view(out.size(0),out.size(1))
        if self.activation:
            out = self.activation(out)
        return out

class MOD_(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, dim=0, n_blk=0, norm='in', activ='relu', init_mode=['normal']):

        super(MOD_, self).__init__()
        # self.model = []
        # self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ, init_mode='kaiming')]
        # for i in range(n_blk - 2):
        #     self.model += [LinearBlock(dim, dim, norm=norm, activation=activ, init_mode='kaiming')]
        # self.model = nn.Sequential(*self.model)

        self.Gen = []
        self.Gen += [ConvBlock(input_dim, output_dim, kernel_size, stride, padding, norm=norm, activation='none', init_mode=init_mode[-1])]
        self.Gen = nn.Sequential(*self.Gen)

    def forward(self, x):
        # x = self.model(x.view(x.size(0), -1))
        x = self.Gen(x)
        return x



def split_caption(caption: str):
    """
    Split the full caption into weather and scene descriptions.
    Expected format:
    [Weather condition], [building quantity/arrangement], [relation to roads or surroundings], 
    [landmarks if visible], [additional layout features if applicable].

    Returns:
        weather_caption: str
        scene_caption: str
    """
    parts = caption.strip().split('.', 1)
    weather = parts[0].strip() if len(parts) > 0 else ""
    scene = parts[1].strip() if len(parts) > 1 else ""
    return weather, scene

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query_feat, context_seq):
        # query_feat: (B, 1, D)
        # context_seq: (B, T, D)
        attn_output, _ = self.attn(query_feat, context_seq, context_seq)
        x = self.norm1(query_feat + attn_output)
        ffn_output = self.ffn(x)
        output = self.norm2(x + ffn_output)
        return output.squeeze(1)  # shape: (B, D)

class GatedFusion(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor([2.0]))  

    def forward(self, orig_feat, attn_feat):

        gate = torch.sigmoid(self.alpha)      
        return gate * orig_feat + (1-gate) * attn_feat


class Pt_ResNet50(nn.Module):
    def __init__(self, pool='avg', init_model=None, norm='ada-ibn', init_mode=['normal'], btnk=[1,0,1], conv_norm='in'):
        super(Pt_ResNet50, self).__init__()
        # model_ft = models.vgg16_bn(pretrained=True)
        model_ft = models.resnet50(pretrained=True)
        model_ft.layer2[3].relu = nn.Sequential()
        self.pool = pool
        if pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
        self.norm = norm
        self.init_mode = init_mode
        self.btnk = btnk
        if norm == 'spade':
            if btnk[0] == 1:
                self.layer1_0_w = MOD_(512, 32, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
                self.layer1_0_b = MOD_(512, 32, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
            if btnk[1] == 1:
                self.layer1_1_w = MOD_(512, 32, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
                self.layer1_1_b = MOD_(512, 32, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
            if btnk[2] == 1:
                self.layer1_2_w = MOD_(512, 32, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
                self.layer1_2_b = MOD_(512, 32, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
            if btnk[3] == 1:
                self.layer2_0_w = MOD_(512, 64, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
                self.layer2_0_b = MOD_(512, 64, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
            if btnk[4] == 1:
                self.layer2_1_w = MOD_(512, 64, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
                self.layer2_1_b = MOD_(512, 64, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
            if btnk[5] == 1:
                self.layer2_2_w = MOD_(512, 64, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
                self.layer2_2_b = MOD_(512, 64, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
            if btnk[6] == 1:
                self.layer2_3_w = MOD_(512, 64, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
                self.layer2_3_b = MOD_(512, 64, kernel_size=3, stride=1, padding=1, norm=conv_norm, activ='none', init_mode=init_mode)
        else:
            self.layer1_w = MLP(512, 32, 512, 3, norm='none', activ='lrelu')
            self.layer1_b = MLP(512, 32, 512, 3, norm='none', activ='lrelu')
            # self.layer2_w = MLP(512, 64, 512, 3, norm='none', activ='lrelu')
            # self.layer2_b = MLP(512, 64, 512, 3, norm='none', activ='lrelu')
            # self.layer3_w = MLP(512, 128, 512, 3, norm='none', activ='lrelu')
            # self.layer3_b = MLP(512, 128, 512, 3, norm='none', activ='lrelu')

            # self.layer1_1_w = MLP(512, 32, 512, 3, norm='none', activ='lrelu')
            # self.layer1_1_b = MLP(512, 32, 512, 3, norm='none', activ='lrelu')

            self.layer1_2_w = MLP(512, 32, 512, 3, norm='none', activ='lrelu')
            self.layer1_2_b = MLP(512, 32, 512, 3, norm='none', activ='lrelu')
            # self.layer2_3_w = MLP(512, 64, 512, 3, norm='none', activ='lrelu')
            # self.layer2_3_b = MLP(512, 64, 512, 3, norm='none', activ='lrelu')
            # self.layer3_5_w = MLP(512, 128, 512, 3, norm='none', activ='lrelu')
            # self.layer3_5_b = MLP(512, 128, 512, 3, norm='none', activ='lrelu')
            # self.layer3_3_w = MLP(512, 128, 512, 3, norm='none', activ='lrelu')
            # self.layer3_3_b = MLP(512, 128, 512, 3, norm='none', activ='lrelu')

        # classification
        # self.bn = nn.BatchNorm1d(512)
        # init.normal_(self.bn.weight.data, 1.0, 0.02)
        # init.constant_(self.bn.bias.data, 0.0)
        
        # 文本特征编码
        self.text_encoder = SimpleTextEncoder(
            model_name="/xxx/bert-base-uncased",
            output_dim=512
        )

        # 投影头（映射到联合嵌入空间）
        self.img_projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        )
        self.txt_projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
        )

        self.weather_modulator = nn.Linear(512, 1)

        classifier = []
        classifier += [nn.BatchNorm1d(512)]
        classifier += [nn.Dropout(p=0.5)]
        # classifier += [nn.Linear(512, 10)]
        classifier += [nn.Linear(512, 11)]
        self.classifier = nn.Sequential(*classifier)
        self.classifier.apply(weights_init_classifier)

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool


    def forward(self, x, scene_captions=None, weather_captions=None):
        # x = self.model.features(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x0 = self.model.maxpool(x)
        x1 = self.model.layer1(x0)
        x2 = self.model.layer2(x1)
        if self.pool == 'avg':
            x = self.model.avgpool2(x2)
            x3 = self.model.avgpool2(x2)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x2)
            x3 = self.model.avgpool2(x2)
            x = x.view(x.size(0), x.size(1))

        img_feat = x3.flatten(1)
        img_emb = self.img_projection(img_feat)  # [B,512]


        txt_emb = None
        weather_logits = None
        mod_factor = None

        if weather_captions is not None:

            w_feat = self.text_encoder(weather_captions)     # [B,512]
            weather_logits = self.classifier(w_feat) # [B, num_weather_classes]

            mod_factor = torch.sigmoid(self.weather_modulator(w_feat))

        if scene_captions is not None:
            # scene_captions: List[str]
            scene_feats = self.text_encoder(scene_captions)     # [B,512]
            txt_emb = self.txt_projection(scene_feats)          # [B,512]

        
        if self.norm == 'spade':
            B, C, H, W = x2.size()
            x2_ = F.interpolate(x2, size=(2*H, 2*W), mode='nearest')

            



            if self.btnk[0] == 1:
                w1 = self.layer1_0_w(x2_)
                b1 = self.layer1_0_b(x2_)
                if mod_factor is not None:
                    mf = mod_factor.view(-1,1,1,1)
                    w1 = w1 * (1+mf)
                    b1 = b1 * (1+mf)
            else:
                w1 = b1 = None
            if self.btnk[1] == 1:
                w1_1 = self.layer1_1_w(x2_)
                b1_1 = self.layer1_1_b(x2_)
                if mod_factor is not None:
                    mf = mod_factor.view(-1,1,1,1)
                    w1_1 *= (1+mf)
                    b1_1 *= (1+mf)
            else:
                w1_1 = b1_1 = None
            if self.btnk[2] == 1:
                w1_2 = self.layer1_2_w(x2_)
                b1_2 = self.layer1_2_b(x2_)
                if mod_factor is not None:
                    mf = mod_factor.view(-1,1,1,1)
                    w1_2 *= (1+mf)
                    b1_2 *= (1+mf)
            else:
                w1_2 = b1_2 = None
            if self.btnk[3] == 1:
                w2_0 = self.layer2_0_w(x2_)
                b2_0 = self.layer2_0_b(x2_)
                if mod_factor is not None:
                    mf = mod_factor.view(-1,1,1,1)
                    w2_0 *= (1+mf)
                    b2_0 *= (1+mf)
            else:
                w2_0 = b2_0 = None
            if self.btnk[4] == 1:
                w2_1 = self.layer2_1_w(x2)
                b2_1 = self.layer2_1_b(x2)
                if mod_factor is not None:
                    mf = mod_factor.view(-1,1,1,1)
                    w2_1 *= (1+mf)
                    b2_1 *= (1+mf)
            else:
                w2_1 = b2_1 = None
            if self.btnk[5] == 1:
                w2_2 = self.layer2_2_w(x2)
                b2_2 = self.layer2_2_b(x2)
                if mod_factor is not None:
                    mf = mod_factor.view(-1,1,1,1)
                    w2_2 *= (1+mf)
                    b2_2 *= (1+mf)
            else:
                w2_2 = b2_2 = None
            if self.btnk[6] == 1:
                w2_3 = self.layer2_3_w(x2)
                b2_3 = self.layer2_3_b(x2)
                if mod_factor is not None:
                    mf = mod_factor.view(-1,1,1,1)
                    w2_3 *= (1+mf)
                    b2_3 *= (1+mf)
            else:
                w2_3 = b2_3 = None
        else:
            w1 = self.layer1_w(x)
            b1 = self.layer1_b(x)
            # w2 = self.layer2_w(x)
            # b2 = self.layer2_b(x)
            # w3 = self.layer3_w(x)
            # b3 = self.layer3_b(x)
            # w1_1 = self.layer1_1_w(x)
            # b1_1 = self.layer1_1_b(x)
            w1_2 = self.layer1_2_w(x)
            b1_2 = self.layer1_2_b(x)
            # w2_3 = self.layer2_3_w(x)
            # b2_3 = self.layer2_3_b(x)
            # w3_5 = self.layer3_5_w(x)
            # b3_5 = self.layer3_5_b(x)
            # w3_3 = self.layer3_3_w(x)
            # b3_3 = self.layer3_3_b(x)
            w2 = None
            b2 = None
            w3 = None
            b3 = None

            w1_1 = None
            b1_1 = None
            w2_3 = None
            b2_3 = None
            w3_5 = None
            b3_5 = None
            w3_3 = None
            b3_3 = None
        # weather classification
        out_w = self.classifier(x)
        if self.norm == 'spade':
            return [[[w1, b1], [w1_1, b1_1], [w1_2, b1_2]], [[w2_0, b2_0], [w2_1,b2_1], [w2_2,b2_2], [w2_3, b2_3]]], out_w, img_emb, txt_emb
        return [w1, w1_1, w1_2], [b1, b1_1, b1_2], [w2, w2_3], [b2, b2_3], [w3, w3_3, w3_5], [b3, b3_3, b3_5], out_w, img_emb, txt_emb





# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', norm='bn', adain='a'):
        super(ft_net, self).__init__()
        if norm == 'bn':
            model_ft = models.resnet50(pretrained=True)
        elif norm == 'ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        elif norm == 'ada-ibn':
            model_ft = resnet50_adaibn_a(pretrained=True, adain=adain)
            # self.pt_model = Pt_ResNet50()
        self.norm = norm
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x

class ft_net_spade(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg', norm='bn', adain='a'):
        super(ft_net_spade, self).__init__()
        if norm == 'bn':
            model_ft = models.resnet50(pretrained=True)
        elif norm == 'ibn':
            model_ft = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        elif norm == 'ada-ibn':
            model_ft = resnet50_adaibn_a(pretrained=True, adain=adain)
            # self.pt_model = Pt_ResNet50()
        elif norm == 'spade':
            model_ft = resnet50_adaibn_a(pretrained=True, adain=adain)
        self.norm = norm
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x, mod_f):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if self.norm == 'spade':
            # print('---------------- spade norm-----------------------')
            x = extract_spade_feature(self.model.layer1, x, mod_f[0])
        else:
            x = self.model.layer1(x)
        if self.norm == 'spade':
            x = extract_spade_feature(self.model.layer2, x, mod_f[1])
        else:
            x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # print(x.size())
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x

# For cvusa/cvact
class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 1, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=2, norm='bn', adain='a', btnk=[1,0,1]):
        super(two_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        self.norm = norm
        self.adain = adain
        self.sqr = True # if the satellite image is square ring partition and the ground image is row partition, self.sqr is True. Otherwise it is False.
        if VGG16:
            if LPN:
                # satelite
                self.model_1 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block = block)
                if self.sqr:
                    self.model_1 = ft_net_VGG16_LPN_R(class_num, stride=stride, pool=pool, block=block)
            else:
                self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
                # self.vgg1 = models.vgg16_bn(pretrained=True)
                # self.model_1 = SAFA()
                # self.model_1 = SAFA_FC(64, 32, 8)
        else:
            #resnet50 LPN cvusa/cvact
            if LPN:
                self.model_1 =  ft_net_cvusa_LPN(class_num, stride=stride, pool = pool, block=block)
                if self.sqr:
                    self.model_1 = ft_net_cvusa_LPN_R(class_num, stride=stride, pool=pool, block=block)
                self.block = self.model_1.block
            else:
                if norm == 'spade':
                    self.model_1 = ft_net_spade(class_num, stride=stride, pool=pool, norm=norm, adain=adain)
                else:
                    self.model_1 = ft_net(class_num, stride = stride, pool = pool, norm=norm, adain=adain)
        if share_weight:
            print('------------------share weight-----------------------')
            self.model_2 = self.model_1
        else:
            if VGG16:
                if LPN:
                    #street
                    self.model_2 = ft_net_VGG16_LPN(class_num, stride=stride, pool=pool, block = block, row = self.sqr)
                else:
                    self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
                    # self.vgg2 = models.vgg16_bn(pretrained=True)
                    # self.model_2 = SAFA()
                    # self.model_2 = SAFA_FC(64, 32, 8)
            else:
                if LPN:
                    self.model_2 =  ft_net_cvusa_LPN(class_num, stride = stride, pool = pool, block=block, row = self.sqr)
                else:
                    self.model_2 = ft_net(class_num, stride = stride, pool = pool, norm=norm, adain=adain)
        if LPN:
            if VGG16:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(1024, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(512, class_num, droprate))
            else:
                if pool == 'avg+max':
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(4096, class_num, droprate))
                else:
                    for i in range(self.block):
                        name = 'classifier'+str(i)
                        setattr(self, name, ClassBlock(2048, class_num, droprate))
        else:    
            self.classifier = ClassBlock(2048, class_num, droprate)
            if pool =='avg+max':
                self.classifier = ClassBlock(4096, class_num, droprate)
            if VGG16:
                self.classifier = ClassBlock(512, class_num, droprate)
                # self.classifier = ClassBlock(4096, class_num, droprate, num_bottleneck=512) #safa 情况下
                if pool =='avg+max':
                    self.classifier = ClassBlock(1024, class_num, droprate)
        if norm == 'ada-ibn' or norm == 'spade':
            self.pt_model = Pt_ResNet50(norm=norm, init_mode=['normal'], btnk=btnk)
            self._w1, self._b1, self._w2, self._b2, self._w3, self._b3 = pretrained_in_weight(True)

    def forward(self, x1, x2):
        if self.LPN:
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)
                y1 = self.part_classifier(x1)

            if x2 is None:
                y2 = None
            else:
                x2 = self.model_2(x2)
                y2 = self.part_classifier(x2)
        else:
            if x1 is None:
                y1 = None
            else:
                if self.norm == 'ada-ibn':
                    sw1, sb1, sw2, sb2, sw3, sb3, sout_w = self.pt_model(x1)
                    if self.adain == 'a':
                        assign_adain_params(sw1[0] + self._w1[0], sb1[0] + self._b1[0], self.model_1.model.layer1[0], 32)  # set layer1's ada-ibn
                        # assign_adain_params(sw1[1] + self._w1[1], sb1[1] + self._b1[1], self.model_1.model.layer1[1], 32)
                        assign_adain_params(sw1[2] + self._w1[2], sb1[2] + self._b1[2], self.model_1.model.layer1[2], 32)
                        # assign_adain_params(sw2[0] + self._w2[0], sb2[0] + self._b2[0], self.model_1.model.layer2[0], 64)  # set layer2's ada-ibn
                        # assign_adain_params(sw2[1] + self._w2[1], sb2[1] + self._b2[1], self.model_1.model.layer2[3], 64)
                        # assign_adain_params(sw3[0] + self._w3[0], sb3[0] + self._b3[0], self.model_1.model.layer3[0], 128)  # set layer3's ada-ibn
                        # assign_adain_params(sw3[1] + self._w3[1], sb3[1] + self._b3[1], self.model_1.model.layer3[3], 128)
                        # assign_adain_params(sw3[2] + self._w3[2], sb3[2] + self._b3[2], self.model_1.model.layer3[5], 128)
                # x1 = self.vgg1.features(x1)
                    x1 = self.model_1(x1)
                elif self.norm == 'spade':
                    smod_f, sout_w = self.pt_model(x1)
                    x1 = self.model_1(x1, smod_f)
                else:
                    x1 = self.model_1(x1)
                y1 = self.classifier(x1)

            if x2 is None:
                y2 = None
            else:
                if self.norm == 'ada-ibn':
                    gw1, gb1, gw2, gb2, gw3, gb3, gout_w = self.pt_model(x2)
                    if self.adain == 'a':
                        assign_adain_params(gw1[0] + self._w1[0], gb1[0] + self._b1[0], self.model_2.model.layer1[0], 32)  # set layer1's ada-ibn
                        # assign_adain_params(gw1[1] + self._w1[1], gb1[1] + self._b1[1], self.model_2.model.layer1[1], 32)
                        assign_adain_params(gw1[2] + self._w1[2], gb1[2] + self._b1[2], self.model_2.model.layer1[2], 32)
                        # assign_adain_params(gw2[0] + self._w2[0], gb2[0] + self._b2[0], self.model_2.model.layer2[0], 64)  # set layer2's ada-ibn
                        # assign_adain_params(gw2[1] + self._w2[1], gb2[1] + self._b2[1], self.model_2.model.layer2[3], 64)
                        # assign_adain_params(gw3[0] + self._w3[0], gb3[0] + self._b3[0], self.model_2.model.layer3[0], 128) # set layer3's ada-ibn
                        # assign_adain_params(dw3[1] + self._w3[1], db3[1] + self._b3[1], self.model_3.model.layer3[3], 128)
                        # assign_adain_params(gw3[2] + self._w3[2], gb3[2] + self._b3[2], self.model_2.model.layer3[5], 128)
                # x2 = self.vgg2.features(x2)
                    x2 = self.model_2(x2)
                elif self.norm == 'spade':
                    gmod_f, gout_w = self.pt_model(x2)
                    x2 = self.model_2(x2, gmod_f)
                else:
                    x2 = self.model_2(x2)
                y2 = self.classifier(x2)
        if self.norm == 'ada-ibn' or self.norm == 'spade':
            if not self.training:
                return y1, y2
            else:
                return y1, y2, sout_w, gout_w
        return y1, y2

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            # part[i] = torch.squeeze(x[:,:,i])
            part[i] = x[:,:,i].view(x.size(0),-1)
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


proj_root = os.path.abspath(os.path.dirname(__file__)) 
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)


xvlm_root = './XVLM/X-VLM-master'
if xvlm_root not in sys.path:
    sys.path.insert(1, xvlm_root)  
from project_utils  import update_average, get_model_list, load_network, save_network, make_weights_for_balanced_classes
from models.xvlm import XVLMBase


class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, LPN=False, block=6, norm='bn', adain='a', circle=False, btnk=[1,0,1], conv_norm='none',config_xvlm=None,,load_text_params=False):
        super(three_view_net, self).__init__()
        self.LPN = LPN
        self.block = block
        self.norm = norm
        self.adain = adain
        self.circle = circle

        assert config_xvlm is not None
        self.xvlm = XVLMBase(
            config=config_xvlm,
            load_vision_params=True,
            load_text_params=False,
            use_contrastive_loss=True,
            use_matching_loss=True,
            use_mlm_loss=config_xvlm.get('use_mlm_loss', True),
            use_bbox_loss=config_xvlm.get('use_bbox_loss', True)
        )
        self.xvlm.config = config_xvlm
        self.vision_encoder = self.xvlm.vision_encoder
        self.text_encoder   = self.xvlm.text_encoder
        self.vision_proj    = self.xvlm.vision_proj
        self.text_proj      = self.xvlm.text_proj
        self.temp           = self.xvlm.temp
        self.itm_head       = self.xvlm.itm_head
        if config_xvlm.get('use_roberta', False):
            self.tokenizer = RobertaTokenizer.from_pretrained(config_xvlm['text_encoder'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config_xvlm['text_encoder'])

        in_feat = config_xvlm['embed_dim']
        self.classifier = ClassBlock(in_feat, class_num, droprate, return_f=circle)
        self.fusion = GatedFusion(feat_dim=in_feat)

    def forward(self, x1, x2, x3, x4 = None, captions=None): # x4 is extra data
        text_feat = None
        cap_inputs = None
        txt_embeds = None
        if captions is not None and self.training:

            cap_inputs = self.tokenizer(
                captions,
                padding='longest',
                truncation=True,
                max_length=self.xvlm.config['max_tokens'],
                return_tensors="pt"
            ).to(x1.device)
            txt_out = self.text_encoder(
                cap_inputs.input_ids,
                attention_mask=cap_inputs.attention_mask,
                return_dict=True,
                mode='text',
                output_hidden_states=True
            )
            txt_embeds = txt_out.hidden_states[-1]   # [B, T, H]
            text_feat = F.normalize(self.text_proj(txt_embeds[:, 0, :]), dim=-1)  # [B, D]


        def run_one(img):
   
            img_embeds, img_atts = self.xvlm.get_vision_embeds(img)          # [B, 1+P, C], [B,1+P]
            img_cls = img_embeds[:, 0, :]                             # [B, C]
            img_feat = F.normalize(self.vision_proj(img_cls), dim=-1)# [B, D]

        
            if self.training and text_feat is not None:
                
                loss_itc = self.xvlm.get_contrastive_loss(img_feat, text_feat)
                
                loss_itm = self.xvlm.get_matching_loss(
                    img_embeds, img_atts,
                    img_feat,
                    txt_embeds, cap_inputs.attention_mask,
                    text_feat
                )
                self.loss_itc = loss_itc
                self.loss_itm = loss_itm

            cls_feat = img_feat
            if self.training and use_text and text_feat is not None:
                cls_feat = F.normalize(self.fusion(img_feat, text_feat), dim=-1)
            out = self.classifier(img_feat)  # [B, class_num]
            return out


        y1 = None if x1 is None else run_one(x1)
        y2 = None if x2 is None else run_one(x2)
        y3 = None if x3 is None else run_one(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            y4 = run_one(x4)
            return y1, y2, y3, y4

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:,:,i].view(x.size(0),-1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y





'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
#     net = two_view_net(701, droprate=0.5, pool='avg', stride=1, VGG16=False, LPN=True, block=8)
    os.environ['CUDA_VISIBLE_DEVICES'] ='1'
    net = three_view_net(701, droprate=0.75, pool='avg', stride=1, share_weight=True, LPN=False, block=3, norm='spade', btnk=[1,1,1,1,1,1,1])
    # net.eval()


    # net = ft_net_VGG16_LPN_R(701)
    # net = ft_net_cvusa_LPN(701, stride=1)
    # net = ft_net(701)
    # net = Pt_ResNet50()
    print(net)
    net.cuda()
    # input = Variable(torch.FloatTensor(2, 3, 256, 256)).cuda()
    input = Variable(torch.randn(2, 3, 256, 256)).cuda()
    # output1,output2 = net(input,input)
    output1,output2,output3,output4,f1,f2 = net(input,input,input,input)
    # output1 = net(input)
    print('net output size:')
    print(output1.shape)
    # print(output.shape)
    # for i in range(len(output1)):
    #     print(output1[i].shape)
    # x = torch.randn(2,512,8,8)
    # x_shape = x.shape
    # pool = AzimuthPool2d(x_shape, 8)
    # out = pool(x)
    # print(out.shape)
    # layer1_w = MLP(512, 96, 512, 3, norm='none', activ='lrelu')
    # print(layer1_w)
