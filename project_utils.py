import os
import json
import torch
import yaml
import torch.nn as nn  # may be used by callers
from ruamel.yaml import YAML

# ---------------------------------------
# Class-balancing utility
# ---------------------------------------
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for _, cls in images:
        count[cls] += 1
    N = float(sum(count))
    # avoid div-by-zero if some class is empty
    weight_per_class = [ (N / c if c > 0 else 0.0) for c in count ]
    weight = [ weight_per_class[cls] for _, cls in images ]
    return weight

# ---------------------------------------
# Model file helpers
# ---------------------------------------
def get_model_list(dirname, key):
    """Return latest checkpoint path containing `key` and ending with .pth"""
    if not os.path.exists(dirname):
        print(f'no dir: {dirname}')
        return None
    gen_models = [
        os.path.join(dirname, f) for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and f.endswith(".pth")
    ]
    if not gen_models:
        print(f'no checkpoints matching key="{key}" under {dirname}')
        return None
    gen_models.sort()
    return gen_models[-1]

def save_network(network, dirname, epoch_label):
    outdir = os.path.join('./model', dirname)
    os.makedirs(outdir, exist_ok=True)
    save_filename = f'net_{epoch_label:03d}.pth' if isinstance(epoch_label, int) else f'net_{epoch_label}.pth'
    save_path = os.path.join(outdir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()
    return save_path

# ---------------------------------------
# Load for resume
# ---------------------------------------
def _safe_assign(opt, cfg, key, default=None):
    """Set opt.key from cfg if present; otherwise keep current or use default."""
    setattr(opt, key, cfg.get(key, getattr(opt, key, default)))

def _build_xvlm_config_from_opt(opt):
    """Build X-VLM config dict safely (no硬编码本地路径)."""
    cfg = {}
    # vision config
    if getattr(opt, 'xvlm_config', None) and os.path.isfile(opt.xvlm_config):
        with open(opt.xvlm_config, 'r') as f:
            cfg = json.load(f)
        cfg['use_swin']      = getattr(opt, 'use_swin', False) or True  # keep previous behavior
        cfg['use_clip_vit']  = getattr(opt, 'use_clip_vit', False)
        cfg['vision_config'] = opt.xvlm_config
        cfg['image_res']     = getattr(opt, 'h', 224)
        cfg['patch_size']    = cfg.get('patch_size', 32)
    else:
        # if no vision config, return minimal cfg so downstream can still pass it
        return {
            'use_swin': True,
            'use_clip_vit': getattr(opt, 'use_clip_vit', False),
            'vision_config': '',
            'image_res': getattr(opt, 'h', 224),
            'patch_size': 32,
            'embed_dim': getattr(opt, 'embed_dim', 256),
            'temp': getattr(opt, 'temp', 0.07),
            'max_tokens': getattr(opt, 'max_tokens', 256),
            'use_mlm_loss': getattr(opt, 'use_mlm_loss', True),
            'use_bbox_loss': getattr(opt, 'use_bbox_loss', True),
            'use_spatial_loss': getattr(opt, 'use_spatial_loss', False),
            'use_roberta': getattr(opt, 'use_roberta', False),
            'text_config': '',
            'text_encoder': 'roberta-base' if getattr(opt, 'use_roberta', False) else 'bert-base-uncased',
        }

    # text side (json/yaml or fallback HF name)
    text_cfg_path = getattr(opt, 'xvlm_text_config', '')
    if text_cfg_path and os.path.isfile(text_cfg_path):
        if text_cfg_path.endswith('.json'):
            _ = json.load(open(text_cfg_path, 'r'))
        else:
            YAML(typ='safe').load(open(text_cfg_path, 'r'))
        cfg['use_roberta']  = getattr(opt, 'use_roberta', False)
        cfg['text_config']  = text_cfg_path
        # Prefer env var to local path, fallback to HF model name
        bert_base_dir = os.environ.get('BERT_BASE_DIR', '').strip()
        if bert_base_dir and os.path.isdir(bert_base_dir):
            cfg['text_encoder'] = bert_base_dir
        else:
            cfg['text_encoder'] = 'roberta-base' if getattr(opt, 'use_roberta', False) else 'bert-base-uncased'
    else:
        cfg['use_roberta']  = getattr(opt, 'use_roberta', False)
        cfg['text_config']  = ''
        cfg['text_encoder'] = 'roberta-base' if getattr(opt, 'use_roberta', False) else 'bert-base-uncased'

    # heads / loss toggles
    cfg['embed_dim']      = getattr(opt, 'embed_dim', 256)
    cfg['temp']           = getattr(opt, 'temp', 0.07)
    cfg['max_tokens']     = getattr(opt, 'max_tokens', 256)
    cfg['use_mlm_loss']   = getattr(opt, 'use_mlm_loss', True)
    cfg['use_bbox_loss']  = getattr(opt, 'use_bbox_loss', True)
    cfg['use_spatial_loss']= getattr(opt, 'use_spatial_loss', False)
    return cfg

def load_network(name, opt):
    """Return (network, opt, epoch) for resume training."""
    from model import ft_net, two_view_net, three_view_net  # keep original import semantics

    dirname = os.path.join('./model', name)
    last_path = get_model_list(dirname, 'net')
    if last_path is None:
        raise FileNotFoundError(f'No checkpoint found under {dirname}')

    last_model_name = os.path.basename(last_path)
    epoch_part = last_model_name.split('_')[1].split('.')[0]
    epoch = int(epoch_part) if epoch_part != 'last' and epoch_part.isdigit() else epoch_part

    config_path = os.path.join(dirname, 'opts.yaml')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'opts.yaml not found under {dirname}')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}

    # ---- sync essential opts (safe-get) ----
    keys = [
        'name','data_dir','train_all','droprate','color_jitter','batchsize','h','w','share',
        'stride','LPN','norm','adain','pool','gpu_ids','erasing_p','lr','use_dense','fp16',
        'views','block','btnk','conv_norm','use_res101','use_VIT','use_vgg','use_vgg16',
        'xvlm_config','xvlm_text_config','use_swin','use_clip_vit','use_roberta'
    ]
    for k in keys:
        _safe_assign(opt, config, k, getattr(opt, k, None))

    # nclasses: prefer config, otherwise keep existing or fallback 701 (legacy)
    opt.nclasses = config.get('nclasses', getattr(opt, 'nclasses', 701))

    # X-VLM combined config (no local hardcoding)
    config_xvlm = _build_xvlm_config_from_opt(opt)

    # ---- build model by views ----
    if opt.views == 3:
        if opt.LPN:
            model = three_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool,
                                   share_weight=opt.share, LPN=True, block=opt.block,
                                   norm=opt.norm, btnk=opt.btnk)
        else:
            if getattr(opt, 'use_res101', False):
                model = three_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool,
                                       share_weight=opt.share, norm=opt.norm, adain=opt.adain,
                                       btnk=opt.btnk, conv_norm=opt.conv_norm, VGG16=getattr(opt, 'use_vgg', False),
                                       Dense=getattr(opt, 'use_dense', False), ResNet101=True)
            elif getattr(opt, 'use_VIT', False):
                model = three_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool,
                                       share_weight=opt.share, norm=opt.norm, adain=opt.adain,
                                       btnk=opt.btnk, conv_norm=opt.conv_norm, VGG16=getattr(opt, 'use_vgg', False),
                                       Dense=getattr(opt, 'use_dense', False), VIT=True)
            else:
                model = three_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool,
                                       share_weight=opt.share, norm=opt.norm, adain=opt.adain,
                                       btnk=opt.btnk, conv_norm=opt.conv_norm, VGG16=getattr(opt, 'use_vgg', False),
                                       config_xvlm=config_xvlm, load_text_params=False)
    else:
        # views == 2
        if getattr(opt, 'use_vgg16', False):
            model = two_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool,
                                 share_weight=opt.share, VGG16=True, norm=opt.norm, adain=opt.adain, btnk=opt.btnk)
            if opt.LPN:
                model = two_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool,
                                     share_weight=opt.share, VGG16=True, LPN=True, block=opt.block)
        else:
            model = two_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool,
                                 share_weight=opt.share, norm=opt.norm, adain=opt.adain, btnk=opt.btnk)

    # ---- load weights ----
    print(f'Load the model from {last_path}')
    network = model
    network_dict = network.state_dict()
    trained_dict = torch.load(last_path, map_location='cpu')

    # Show missing/unexpected keys more clearly
    missing = [k for k in network_dict.keys() if k not in trained_dict]
    unexpected = [k for k in trained_dict.keys() if k not in network_dict]
    if missing:
        print(f'[load_network] Missing keys: {len(missing)} (showing up to 10): {missing[:10]}')
    if unexpected:
        print(f'[load_network] Unexpected keys in checkpoint: {len(unexpected)} (showing up to 10): {unexpected[:10]}')

    # strict=False style loading
    trained_filtered = {k: v for k, v in trained_dict.items() if k in network_dict}
    network_dict.update(trained_filtered)
    network.load_state_dict(network_dict)
    return network, opt, epoch

# ---------------------------------------
# EMA utilities
# ---------------------------------------
def toogle_grad(model, requires_grad):
    """Keep name for backward-compat (typo preserved)."""
    for p in model.parameters():
        p.requires_grad_(requires_grad)

# alias with the correct spelling to avoid future confusion
def toggle_grad(model, requires_grad):
    toogle_grad(model, requires_grad)

def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    src_params = dict(model_src.named_parameters())
    for name, p_tgt in model_tgt.named_parameters():
        p_src = src_params[name]
        assert p_src is not p_tgt
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    toogle_grad(model_src, True)
