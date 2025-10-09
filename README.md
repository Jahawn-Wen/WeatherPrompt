# WeatherPrompt


**WeatherPrompt** is a **training-free, all-weather, text-guided cross-view localization framework**. The core idea is to introduce **open-ended weather descriptions** into the retrieval process to counteract the observational shifts and cross-view misalignments caused by rain, fog, snow, and night conditions.

We introduce a **training-free multi-weather prompting pipeline**: leveraging University-1652 and SUES-200, we synthesize adverse-weather views and use a **two-stage prompt (“weather-first, then spatial”)** to express **coarse-to-fine semantics**, thereby **auto-generating high-quality open-set image–text pairs** for alignment and supervision.

Extensive experiments validate the effectiveness of WeatherPrompt. On University-1652, Drone $\rightarrow$ Satellite achieves R@1 77.14\% / AP 80.20\% and Satellite $\rightarrow$ Drone achieves R@1 87.72\% / AP 76.39\%. Compared with representative methods, Drone $\rightarrow$ Satellite improves R@1 by 11.99\% and AP by 11.04\%, while Satellite $\rightarrow$ Drone improves R@1 by 3.04\% and AP by 10.64\%. On SUES-200, Satellite $\rightarrow$ Drone reaches R@1 80.73\% / AP 66.12\%, showing consistent advantages. Under real-world Dark+Rain+Fog videos, Drone $\rightarrow$ Satellite attains AP 64.94\% and Satellite $\rightarrow$ Drone AP 72.15\%, evidencing strong generalization and robustness in adverse weather. More details can be found at our paper: [WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization](https://arxiv.org/pdf/2508.09560)


## News
* The **CoT Prompt** is released, Welcome to communicate！
* The **Models** and **Weights** are released, Welcome to communicate！
* We provide some of the tools in the **Weather.py**.


## CoT Prompt
* The **prompt** format:
```
Given an aerial image. Based only on the image, generate a concise and truthful description (target length 100–120 characters; if this is hard to meet, prioritize accuracy and do not pad), avoiding any speculation. Follow these steps:

1. Overall assessment: Observe the sky, lighting, and color tone to determine the image’s overall atmosphere. Based solely on these visual cues, describe the primary weather impression.
2. Local detail analysis: Look for specific evidence such as raindrops, fog, snowflakes, shadow changes, reflections, or any visual cues indicating weather effects.
3. Weather inference: Based on your comprehensive and detailed observations, infer the specific weather condition. Clearly state the weather you observe.
4. Describe visible structures (buildings, roads, open spaces): their quantities, arrangement, and spatial relationships.
5. Do not infer or guess any elements that are not visible.
6. Output format: [Weather description], [Building layout], [Landmarks (if visible)], [Relation to roads or surroundings], [Other layout features (if applicable)].
``` 

## Open-Weather Description
* We utilize the [imgaug](https://github.com/aleju/imgaug) library to synthetically realistic weather variations.
* We randomly select only one drone-view image per region as a representative.
* We apply a pretrained large multimodal model [Qwen2.5-VL-32B](https://qwen.ai/research) for automatic weather description through CoT Prompt.

Organize `dataset` folder as follows:

```
|-- dataset/
|    |-- University-Release/
|        |-- test/
|            |-- query_drone/
|            |-- query_satellite/
|            |-- ...
|        |-- train/
|            |-- drone/
|            |-- satellite/
|            |-- ...
|    |-- SUES/
|        |-- Training/
|            |-- 150/
|            |-- 200/
|            |-- ...
|        |-- Testing/
|            |-- 150/
|            |-- 200/
|            |-- ...
|    |-- multiweather_captions_32B.json
|    |-- multiweather_captions_test_32B.json
|    |-- multiweather_captions_test_32B_gallery.json
```


## Models and Weights
*  The **Models** and **Weights** are released.
* Download The Trained Model Weights:[Baidu Yun](https://pan.baidu.com/s/1bvu80h-GJ-s0Cffyk2pqbA?pwd=3wjy)[3wjy]

Organize `XVLM` folder as follows:

```
|-- XVLM/
|    |-- X-VLM-master/
|        |-- accelerators/
|        |-- configs/
|        |-- ...
|        |-- Captioning_pretrain.py
|        |-- ...
|    |-- 4m_base_model_state_step_199999.th
|    |-- 16m_base_model_state_step_199999.th
|-- image_folder.py/
|-- ...
```



## Usage
### Install Requirements

We use single A6000 48G GPU for training and evaluation.

Create conda environment.

```
conda create -n weatherprompt python=3.9
conda activate weatherprompt
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install -r requirements.txt
```

### Datasets Prepare
Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request. You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

Download [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark).

## Train & Evaluation
### Train & Evaluation University-1652/SUES-200 (Change the dataset path)
```  
sh run.sh
sh run_test.sh
```

## Reference

```bibtex
@inproceedings{wen2025WeatherPrompt,
  author = "Wen, Jiahao and Yu, Hang and Zheng, Zhedong",
  title = "WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization",
  booktitle = "NeurIPS",
  year = "2025" }
```

## ✨ Acknowledgement
- Our code is based on [XVLM](https://github.com/zengyan-97/X-VLM)
- [Qwen2.5-VL-32B](https://qwen.ai/research): Thanks a lot for the foundamental efforts!




