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


