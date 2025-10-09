# train spade net
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_1_spade_210ep_weather_0110000_fin_reptile5task_auto_scope' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.005_1_spade_210ep_weather_0110000_fin_reptile5task_auto_scope' \
# --data_dir='/home/wjh/project/MuseNet-master-test/dataset/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.005 \
# --gpu_ids='2' \
# --norm='spade' \
# --iaa \
# --multi_weather \
# --btnk 0 1 1 0 0 0 0 \
# --conv_norm='none' \
# --adain='a'



# train spade net (reptile)
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.006_spade_210ep_weather_0110000_fin_reptile5task_auto_scope' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.006_spade_210ep_weather_0110000_fin_reptile5task_auto_scope' \
# --data_dir='/home/wjh/project/MuseNet-master-test/dataset/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.006 \
# --gpu_ids='3' \
# --norm='spade' \
# --iaa \
# --multi_weather \
# --btnk 0 1 1 0 0 0 0 \
# --conv_norm='none' \
# --reptile \
# --adain='a'

# train vit
# python train.py \
# --name='qwen5.3_w_b_xvlm_focal_cl+reptile_7B_chain_new_v1_spade1' \
# --experiment_name='qwen5.3_w_b_xvlm_focal_cl+reptile_7B_chain_new_v1_spade1' \
# --data_dir='/home/wjh/project/MuseNet-master-test/dataset/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=224 \
# --w=224 \
# --lr=0.005 \
# --gpu_ids='1' \
# --norm='spade' \
# --reptile \
# --qwen \
# --iaa \
# --focal \
# --multi_weather \
# --btnk 0 1 1 0 0 0 0 \
# --conv_norm='none' \
# --adain='a'

# XVLM-B-32
# python train.py \
# --name='qwen5.13_xvlm_focal_reptile_32B_spade_sues' \
# --experiment_name='qwen5.13_xvlm_focal_reptile_32B_spade_sues' \
# --data_dir='/home/wjh/project/MuseNet-master-test_sam/dataset/2/SUES-200-512x512/Training/150' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=384 \
# --w=384 \
# --lr=0.005 \
# --data_dir_grd="/home/wjh/project/MuseNet-master-test_sam/dataset/University-Release/train" \
# --gpu_ids='0' \
# --norm='spade' \
# --reptile \
# --qwen \
# --iaa \
# --focal \
# --multi_weather \
# --btnk 0 1 1 0 0 0 0 \
# --conv_norm='none' \
# --adain='a'

python train.py \
--name='qwen8.5_xvlm_focal_reptile_32B_spade_ft_snow' \
--experiment_name='qwen8.5_xvlm_focal_reptile_32B_spade_ft_snow' \
--data_dir='/home/wjh/project/MuseNet-master-test/dataset/University-Release/train' \
--views=3 \
--droprate=0.5 \
--extra \
--share \
--stride=1 \
--h=384 \
--w=384 \
--lr=0.005 \
--gpu_ids='0' \
--norm='spade' \
--reptile \
--qwen \
--iaa \
--focal \
--multi_weather \
--btnk 0 1 1 0 0 0 0 \
--conv_norm='none' \
--adain='a'


# python test_iaa_caption.py \
# --name='qwen5.3_w_b_xvlm_focal_cl+reptile_7B_chain_new_v1_spade1' \
# --test_dir='/home/wjh/project/MuseNet-master-test/dataset/University-Release/test' \
# --batchsize=128 \
# --iaa \
# --gpu_ids='1'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_spade_210ep_weather_0110000_fin_5' \
# --test_dir='/home/wjh/project/MuseNet-master/dataset/University-Release/test' \
# --gpu_ids='1'


# training ibn net
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_ibn_v24.11_210ep_weather_reptile1' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.005_ibn_v24.11_210ep_weather_reptile1' \
# --data_dir='/home/wjh/project/MuseNet-master-test/dataset/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.005 \
# --gpu_ids='3' \
# --norm='ibn' \
# --iaa \
# --multi_weather \
# --btnk 1 0 0 0 0 0 0 \
# --conv_norm='none' \
# --reptile \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_ibn_v24.11_210ep_weather' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='4'


# train LPN weather
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_v24.11_210ep_weather' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_v24.11_210ep_weather' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.001 \
# --gpu_ids='5' \
# --LPN \
# --iaa \
# --multi_weather \
# --block=4 \
# --conv_norm='none' \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_v24.11_210ep_weather' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='5'

# LPN + Spade
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_Spade_v24.11_210ep_weather' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_Spade_v24.11_210ep_weather' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.001 \
# --gpu_ids='0' \
# --LPN \
# --iaa \
# --multi_weather \
# --btnk 0 1 1 0 0 0 0 \
# --norm='spade' \
# --block=4 \
# --conv_norm='none' \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.001_LPN_Spade_v24.11_210ep_weather' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='4'

# train vgg16 weather
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_vgg_v24.11_210ep_weather1' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.005_vgg_v24.11_210ep_weather1' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.005 \
# --gpu_ids='3' \
# --iaa \
# --use_vgg \
# --multi_weather \
# --btnk 1 0 0 0 0 0 0 \
# --conv_norm='none' \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_vgg_v24.11_210ep_weather1' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='3'

# train ResnNet101 weather
# python train.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_Res101_v24.11_210ep_weather' \
# --experiment_name='three_view_long_share_d0.5_256_s1_google_lr0.005_Res101_v24.11_210ep_weather' \
# --data_dir='/home/wangtingyu/datasets/University-Release/train' \
# --views=3 \
# --droprate=0.5 \
# --extra \
# --share \
# --stride=1 \
# --h=256 \
# --w=256 \
# --lr=0.005 \
# --gpu_ids='0' \
# --iaa \
# --use_res101 \
# --multi_weather \
# --btnk 0 0 0 0 0 0 0 \
# --conv_norm='none' \
# --adain='a'

# python test_iaa.py \
# --name='three_view_long_share_d0.5_256_s1_google_lr0.005_Res101_v24.11_210ep_weather' \
# --test_dir='/home/wangtingyu/datasets/University-Release/test' \
# --iaa \
# --gpu_ids='0'
