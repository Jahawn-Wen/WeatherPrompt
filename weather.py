# import os
# import random
# import numpy as np
# from PIL import Image
# from imgaug import augmenters as iaa
# from tqdm import tqdm

# # 1. 定义你的天气增强列表
# iaa_weather_list = [
#     None,
#     iaa.Sequential([
#         iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
#                        intensity_coarse_scale=2, alpha_min=1.0,
#                        alpha_multiplier=0.9, alpha_size_px_max=10,
#                        alpha_freq_exponent=-2, sparsity=0.9,
#                        density_multiplier=0.5, seed=35)
#     ]),
#     iaa.Sequential([
#         iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
#         iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
#         iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
#         iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
#         iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
#     ]),
#     iaa.Sequential([
#         iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
#         iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
#         iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
#         iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
#         iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
#     ]),
#     iaa.Sequential([
#         iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
#         iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991)
#     ]),
#     iaa.Sequential([
#         iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992)
#     ]),
#     iaa.Sequential([
#         iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
#                        intensity_coarse_scale=2, alpha_min=1.0,
#                        alpha_multiplier=0.9, alpha_size_px_max=10,
#                        alpha_freq_exponent=-2, sparsity=0.9,
#                        density_multiplier=0.5, seed=35),
#         iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
#         iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)
#     ]),
#     iaa.Sequential([
#         iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
#                        intensity_coarse_scale=2, alpha_min=1.0,
#                        alpha_multiplier=0.9, alpha_size_px_max=10,
#                        alpha_freq_exponent=-2, sparsity=0.9,
#                        density_multiplier=0.5, seed=35),
#         iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
#         iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)
#     ]),
#     iaa.Sequential([
#         iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
#         iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
#         iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
#         iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
#         iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
#     ]),
#     iaa.Sequential([
#         iaa.MotionBlur(15, seed=17)
#     ])
# ]

# weather_names = ['normal', 'fog', 'rain', 'snow', 'dark', 'light',
#                  'fog_rain', 'fog_snow', 'rain_snow', 'wind']

# drone_root = '/home/wjh/project/WeatherPrompt/dataset/University-Release/test/gallery_drone'
# output_root = '/home/wjh/project/WeatherPrompt/output_weather_test/gallery_drone'
# os.makedirs(output_root, exist_ok=True)

# # 外层进度条：类别遍历
# for class_id in tqdm(os.listdir(drone_root), desc="Processing classes"):
#     class_dir = os.path.join(drone_root, class_id)
#     if not os.path.isdir(class_dir):
#         continue

#     imgs = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     if not imgs:
#         continue
#     chosen = random.choice(imgs)
#     img_np = np.array(Image.open(os.path.join(class_dir, chosen)).convert('RGB'))

#     save_dir = os.path.join(output_root, class_id)
#     os.makedirs(save_dir, exist_ok=True)
#     base, ext = os.path.splitext(chosen)

#     # 内层进度条：天气版本
#     for idx, augmenter in enumerate(tqdm(iaa_weather_list, desc=f"  {class_id} weathers", leave=False)):
#         out_np = img_np if augmenter is None else augmenter(image=img_np)
#         out_pil = Image.fromarray(out_np)
#         out_name = f"{base}-{weather_names[idx]}{ext}"
#         out_pil.save(os.path.join(save_dir, out_name))

# print("All classes processed.")


# SUES200
# import os
# import random
# import numpy as np
# from PIL import Image
# from imgaug import augmenters as iaa
# from tqdm import tqdm

# # —— 1. 定义天气增强列表（保持不变） ——
# iaa_weather_list = [
#     None,
#     iaa.Sequential([iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
#                                    intensity_coarse_scale=2, alpha_min=1.0,
#                                    alpha_multiplier=0.9, alpha_size_px_max=10,
#                                    alpha_freq_exponent=-2, sparsity=0.9,
#                                    density_multiplier=0.5, seed=35)]),
#     iaa.Sequential([
#         iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
#         iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
#         iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
#         iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
#         iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
#     ]),
#     # … 其他增强保持不变 …
# ]
# weather_names = ['normal','fog','rain','snow','dark','light',
#                  'fog_rain','fog_snow','rain_snow','wind']

# # —— 2. 指定训练集根目录和输出目录 ——
# training_root = '/home/wjh/project/WeatherPrompt/dataset/2/SUES-200-512x512/Testing'
# output_root   = '/home/wjh/project/WeatherPrompt/output_weather_SUES_test/gallery_drone'
# os.makedirs(output_root, exist_ok=True)

# # —— 3. 遍历每个子集 —— 
# for subset in tqdm(sorted(os.listdir(training_root)), desc="Processing subsets"):
#     subset_dir = os.path.join(training_root, subset)
#     drone_root = os.path.join(subset_dir, 'gallery_drone')
#     if not os.path.isdir(drone_root):
#         continue

#     # —— 4. 遍历每个 class_id 文件夹 —— 
#     for class_id in tqdm(sorted(os.listdir(drone_root)), desc=f"  {subset} classes", leave=False):
#         class_dir = os.path.join(drone_root, class_id)
#         if not os.path.isdir(class_dir):
#             continue

#         imgs = [f for f in os.listdir(class_dir)
#                 if f.lower().endswith(('.jpg','jpeg','.png'))]
#         if not imgs:
#             continue

#         # —— 随机选一张图片 —— 
#         chosen = random.choice(imgs)
#         img_np = np.array(Image.open(os.path.join(class_dir, chosen))
#                            .convert('RGB'))

#         # —— 为当前子集和 class_id 创建输出目录 —— 
#         save_dir = os.path.join(output_root, subset, class_id)
#         os.makedirs(save_dir, exist_ok=True)
#         base, ext = os.path.splitext(chosen)

#         # —— 对这张图依次应用所有天气增强 —— 
#         for idx, augmenter in enumerate(iaa_weather_list):
#             out_np  = img_np if augmenter is None else augmenter(image=img_np)
#             out_pil = Image.fromarray(out_np)
#             out_name = f"{base}-{weather_names[idx]}{ext}"
#             out_pil.save(os.path.join(save_dir, out_name))

# print("All subsets processed.")



# University-1652
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 配置路径
weather_root = '/home/wjh/project/WeatherPrompt/output_weather_test/gallery_drone'
output_json = '/home/wjh/project/WeatherPrompt/multiweather_captions_test_32B_gallery_new.json'

# 加载 Qwen 模型和处理器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/wjh/project/VLM/Qwen2.5-VL-main/Qwen2.5-VL-32B-int",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained("/home/wjh/project/VLM/Qwen2.5-VL-main/Qwen2.5-VL-32B-int")
processor.tokenizer.padding_side = 'left'

# 构建 prompt
prompt = processor.apply_chat_template(
    [{"role":"user","content":[
        {"type":"image","image":None},
        {"type":"text","text":"**English translation:**Given an aerial image. Based only on the image, generate a concise and truthful description (target length 100–120 characters; if this is hard to meet, prioritize accuracy and do not pad), avoiding any speculation. Follow these steps: 1. Overall assessment: Observe the sky, lighting, and color tone to determine the image’s overall atmosphere. Based solely on these visual cues, describe the primary weather impression. 2. Local detail analysis: Look for specific evidence such as raindrops, fog, snowflakes, shadow changes, reflections, or any visual cues indicating weather effects. 3. Weather inference: Based on your comprehensive and detailed observations, infer the specific weather condition. Clearly state the weather you observe. 4. Describe visible structures (buildings, roads, open spaces): their quantities, arrangement, and spatial relationships. 5. Do not infer or guess any elements that are not visible. 6. Output format: [Weather description], [Building layout], [Landmarks (if visible)], [Relation to roads or surroundings], [Other layout features (if applicable)]."}
    ]}],
    tokenize=False,
    add_generation_prompt=True
)

captions = {}

# 遍历每个类别文件夹（Batch 全类别天气图像）
for class_id in tqdm(os.listdir(weather_root), desc="Classes"):
    class_dir = os.path.join(weather_root, class_id)
    if not os.path.isdir(class_dir):
        continue

    # 收集本类别下所有天气图像
    img_names = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(('.jpg','.jpeg','.png'))
    ])
    if not img_names:
        continue

    # 批量加载 PIL 图像
    images = [Image.open(os.path.join(class_dir, name)).convert("RGB") for name in img_names]

    # 批量编码
    inputs = processor(
        text=[prompt] * len(images),
        images=images,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 一次性生成本类别所有天气 caption
    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=128,
            max_length=inputs["input_ids"].size(1) + 128
        )

    # 去掉 prompt 部分，batch_decode
    input_len = inputs["input_ids"].size(1)
    trimmed = outputs[:, input_len:]
    captions_list = processor.batch_decode(trimmed, skip_special_tokens=True)

    # 存入 captions 字典
    captions[class_id] = {}
    for name, cap in zip(img_names, captions_list):
        weather = name.rsplit('-',1)[-1].rsplit('.',1)[0]
        captions[class_id][weather] = cap

# 写入 JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(captions, f, ensure_ascii=False, indent=2)

print(f"✅ Saved captions to {output_json}")


#SUES200
# import os
# import json
# import torch
# from PIL import Image
# from tqdm import tqdm
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# # —— 修改这两行为你的路径 —— 
# weather_root = '/home/wjh/project/WeatherPrompt/output_weather_SUES_test/query_drone'
# output_json  = '/home/wjh/project/WeatherPrompt/multiweather_captions_test_32B_SUES.json'

# # 加载 Qwen 模型和处理器
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/home/wjh/project/VLM/Qwen2.5-VL-main/Qwen2.5-VL-32B-int",
#     torch_dtype=torch.float16,
#     attn_implementation="flash_attention_2",
#     device_map="auto"
# ).eval()
# processor = AutoProcessor.from_pretrained("/home/wjh/project/VLM/Qwen2.5-VL-main/Qwen2.5-VL-32B-int")
# processor.tokenizer.padding_side = 'left'

# # 构建 prompt（保持不变）
# prompt = processor.apply_chat_template(
#     [{"role":"user","content":[
#         {"type":"image","image":None},
#         {"type":"text","text":"**English translation:**Given an aerial image. Based only on the image, generate a concise and truthful description (target length 100–120 characters; if this is hard to meet, prioritize accuracy and do not pad), avoiding any speculation. Follow these steps: 1. Overall assessment: Observe the sky, lighting, and color tone to determine the image’s overall atmosphere. Based solely on these visual cues, describe the primary weather impression. 2. Local detail analysis: Look for specific evidence such as raindrops, fog, snowflakes, shadow changes, reflections, or any visual cues indicating weather effects. 3. Weather inference: Based on your comprehensive and detailed observations, infer the specific weather condition. Clearly state the weather you observe. 4. Describe visible structures (buildings, roads, open spaces): their quantities, arrangement, and spatial relationships. 5. Do not infer or guess any elements that are not visible. 6. Output format: [Weather description], [Building layout], [Landmarks (if visible)], [Relation to roads or surroundings], [Other layout features (if applicable)]."}
    ]}],
#     tokenize=False,
#     add_generation_prompt=True
# )

# captions = {}

# # —— 双层遍历：先子集，再 class_id —— 
# for subset in tqdm(os.listdir(weather_root), desc="Subsets"):
#     subset_dir = os.path.join(weather_root, subset)
#     if not os.path.isdir(subset_dir):
#         continue

#     captions[subset] = {}

#     # 遍历每个子集下的 class_id 文件夹
#     for class_id in tqdm(os.listdir(subset_dir), desc=f"  {subset} classes", leave=False):
#         class_dir = os.path.join(subset_dir, class_id)
#         if not os.path.isdir(class_dir):
#             continue

#         # 收集本 class_id 下所有天气图
#         img_names = sorted([
#             f for f in os.listdir(class_dir)
#             if f.lower().endswith(('.jpg', '.jpeg', '.png'))
#         ])
#         if not img_names:
#             continue

#         # 批量读取并转换为 PIL.Image
#         images = [
#             Image.open(os.path.join(class_dir, name)).convert("RGB")
#             for name in img_names
#         ]

#         # 批量编码
#         inputs = processor(
#             text=[prompt] * len(images),
#             images=images,
#             padding=True,
#             return_tensors="pt"
#         ).to(device)

#         # 生成 captions
#         with torch.no_grad():
#             outputs = qwen_model.generate(
#                 **inputs,
#                 max_new_tokens=128,
#                 max_length=inputs["input_ids"].size(1) + 128
#             )

#         # 去掉 prompt 部分并解码
#         input_len     = inputs["input_ids"].size(1)
#         trimmed       = outputs[:, input_len:]
#         captions_list = processor.batch_decode(trimmed, skip_special_tokens=True)

#         # 存入三级字典：subset → class_id → weather
#         captions[subset][class_id] = {}
#         for name, cap in zip(img_names, captions_list):
#             # filename like "image-12-dark.jpeg" → weather="dark"
#             weather = name.rsplit('-', 1)[-1].rsplit('.', 1)[0]
#             captions[subset][class_id][weather] = cap

# # 写入 JSON
# with open(output_json, 'w', encoding='utf-8') as f:
#     json.dump(captions, f, ensure_ascii=False, indent=2)

# print(f"✅ Saved captions to {output_json}")







# import json
# from pathlib import Path

# # 你的 JSON 路径
# json_path = Path(
#     "/home/wjh/project/WeatherPrompt/multiweather_captions_test_32B_gallery.json"
# )

# # 1. 读取 JSON
# with json_path.open("r", encoding="utf-8") as f:
#     data = json.load(f)

# def clean_str(s: str) -> str:
#     # 去掉所有 '*'，包括单个和连续的
#     return s.replace("*", "")

# # 2. 清洗
# if isinstance(data, list):
#     # 列表结构：清洗每个 dict 里的 caption
#     for item in data:
#         if isinstance(item, dict) and "caption" in item:
#             item["caption"] = clean_str(item["caption"])

# elif isinstance(data, dict):
#     # 字典结构：可能是 id->inner_dict，也可能是 id->caption_str
#     for key, val in data.items():
#         if isinstance(val, dict):
#             # 内层也是 dict，清洗它的每个字符串字段
#             for sub_key, sub_val in val.items():
#                 if isinstance(sub_val, str):
#                     val[sub_key] = clean_str(sub_val)
#         elif isinstance(val, str):
#             # 直接就是字符串
#             data[key] = clean_str(val)

# else:
#     raise ValueError(f"不支持的 JSON 顶层类型：{type(data)}")

# # 3. 写回文件（覆盖原文件）
# with json_path.open("w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print(f"✅ 已清除所有 '*' 标记并保存到 {json_path}")





# import os
# import io
# import json
# import base64
# from PIL import Image
# from openai import OpenAI

# # -----------------------
# # 配置区
# # -----------------------
# API_KEY    = "XXX"  # 建议改为环境变量读取
# BASE_URL   = "https://api.siliconflow.cn/v1"  # 或者 "https://api.openai.com/v1"
# MODEL_NAME = "THUDM/GLM-4.1V-9B-Thinking"

# # 本地图片根目录：gallery_drone 下每个子文件夹是一个 class_id
# WEATHER_ROOT = "/home/wjh/project/WeatherPrompt/output_weather"
# OUTPUT_JSON  = "/home/wjh/project/WeatherPrompt/multiweather_captions_glm_thinking.json"

# # 通用 prompt
# PROMPT_TEXT = (
#     "**English translation:**Given an aerial image. Based only on the image, generate a concise and truthful description (target length 100–120 characters; if this is hard to meet, prioritize accuracy and do not pad), avoiding any speculation. Follow these steps: 1. Overall assessment: Observe the sky, lighting, and color tone to determine the image’s overall atmosphere. Based solely on these visual cues, describe the primary weather impression. 2. Local detail analysis: Look for specific evidence such as raindrops, fog, snowflakes, shadow changes, reflections, or any visual cues indicating weather effects. 3. Weather inference: Based on your comprehensive and detailed observations, infer the specific weather condition. Clearly state the weather you observe. 4. Describe visible structures (buildings, roads, open spaces): their quantities, arrangement, and spatial relationships. 5. Do not infer or guess any elements that are not visible. 6. Output format: [Weather description], [Building layout], [Landmarks (if visible)], [Relation to roads or surroundings], [Other layout features (if applicable)]."}
# )

# # -----------------------
# # 工具函数：图片转 WebP + Base64
# # -----------------------
# def convert_image_to_webp_base64(image_path: str) -> str:
#     try:
#         with Image.open(image_path) as img:
#             buf = io.BytesIO()
#             img.save(buf, format="WEBP")
#             return base64.b64encode(buf.getvalue()).decode("utf-8")
#     except Exception as e:
#         print(f"[Error] cannot convert {image_path}: {e}")
#         return None

# # -----------------------
# # 初始化 API 客户端
# # -----------------------
# client = OpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
# )

# # -----------------------
# # 主流程
# # -----------------------
# captions = {}

# for class_id in sorted(os.listdir(WEATHER_ROOT)):
#     class_dir = os.path.join(WEATHER_ROOT, class_id)
#     if not os.path.isdir(class_dir):
#         continue

#     captions[class_id] = {}
#     for fname in sorted(os.listdir(class_dir)):
#         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue

#         img_path = os.path.join(class_dir, fname)
#         print(f"Processing {class_id}/{fname} ...")

#         b64 = convert_image_to_webp_base64(img_path)
#         if b64 is None:
#             captions[class_id][fname] = ""
#             continue

#         # 构造消息列表
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b64}"}},
#                     {"type": "text",      "text": PROMPT_TEXT}
#                 ]
#             }
#         ]

#         # 调用接口
#         try:
#             resp = client.chat.completions.create(
#                 model=MODEL_NAME,
#                 messages=messages,
#                 stream=False
#             )
#             text = resp.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"[Error] API call failed for {fname}: {e}")
#             text = ""

#         # 从文件名提取天气标签（如 xxx-rainy.webp → rainy）
#         weather = fname.rsplit("-", 1)[-1].rsplit(".", 1)[0]
#         captions[class_id][weather] = text

# # 将结果写入 JSON
# with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(captions, f, ensure_ascii=False, indent=2)

# print(f"✅ All done! Captions saved to {OUTPUT_JSON}")



# import os
# import io
# import json
# import base64
# from PIL import Image
# from openai import OpenAI
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# # -----------------------
# # 配置区
# # -----------------------
# API_KEY      = os.getenv("OPENAI_API_KEY", "xxx")   # 建议用环境变量
# BASE_URL     = "https://api.siliconflow.cn/v1"         # 或 "https://api.openai.com/v1"
# MODEL_NAME   = "THUDM/GLM-4.1V-9B-Thinking"
# WEATHER_ROOT = "/home/wjh/project/WeatherPrompt/output_weather"
# OUTPUT_JSON  = "/home/wjh/project/WeatherPrompt/multiweather_captions_glm_thinking.json"

# PROMPT_TEXT = (
#   "**English translation:**Given an aerial image. Based only on the image, generate a concise and truthful description (target length 100–120 characters; if this is hard to meet, prioritize accuracy and do not pad), avoiding any speculation. Follow these steps: 1. Overall assessment: Observe the sky, lighting, and color tone to determine the image’s overall atmosphere. Based solely on these visual cues, describe the primary weather impression. 2. Local detail analysis: Look for specific evidence such as raindrops, fog, snowflakes, shadow changes, reflections, or any visual cues indicating weather effects. 3. Weather inference: Based on your comprehensive and detailed observations, infer the specific weather condition. Clearly state the weather you observe. 4. Describe visible structures (buildings, roads, open spaces): their quantities, arrangement, and spatial relationships. 5. Do not infer or guess any elements that are not visible. 6. Output format: [Weather description], [Building layout], [Landmarks (if visible)], [Relation to roads or surroundings], [Other layout features (if applicable)]."}
# )

# # -----------------------
# # 图片转 WebP + Base64 (带缩放 & 质量控制)
# # -----------------------
# def convert_image_to_webp_base64(image_path: str, max_size=512, quality=50) -> str:
#     try:
#         with Image.open(image_path) as img:
#             # 按比例缩放，保证长边不超过 max_size
#             img.thumbnail((max_size, max_size), Image.LANCZOS)
#             buf = io.BytesIO()
#             img.save(buf, format="WEBP", quality=quality)
#             return base64.b64encode(buf.getvalue()).decode("utf-8")
#     except Exception as e:
#         print(f"[Error] convert {image_path}: {e}")
#         return None

# # -----------------------
# # 初始化 API 客户端
# # -----------------------
# client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# # -----------------------
# # 单张图像处理函数
# # -----------------------
# def process_image(class_id: str, fname: str):
#     img_path = os.path.join(WEATHER_ROOT, class_id, fname)
#     b64 = convert_image_to_webp_base64(img_path)
#     if not b64:
#         return class_id, fname, ""
#     messages = [{
#         "role": "user",
#         "content": [
#             {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b64}"}},
#             {"type": "text",      "text": PROMPT_TEXT}
#         ]
#     }]
#     try:
#         resp = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=messages,
#             stream=False
#         )
#         text = resp.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"[Error] API failed {class_id}/{fname}: {e}")
#         text = ""
#     # 从文件名提取天气标签
#     weather = fname.rsplit("-", 1)[-1].rsplit(".", 1)[0]
#     return class_id, weather, text

# # -----------------------
# # 主流程：并发执行
# # -----------------------
# if __name__ == "__main__":
#     # 收集所有任务
#     tasks = []
#     for class_id in sorted(os.listdir(WEATHER_ROOT)):
#         dir_path = os.path.join(WEATHER_ROOT, class_id)
#         if not os.path.isdir(dir_path): continue
#         for fname in sorted(os.listdir(dir_path)):
#             if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                 tasks.append((class_id, fname))

#     captions = {cid: {} for cid, _ in tasks}

#     # 并发 + 进度条
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         # 用 map + tqdm，自动显示已完成/总数和 ETA
#         for cid, weather, text in tqdm(
#             executor.map(lambda args: process_image(*args), tasks),
#             total=len(tasks),
#             desc="Generating captions"
#         ):
#             captions[cid][weather] = text

#     # 写入 JSON
#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(captions, f, ensure_ascii=False, indent=2)

#     print(f"✅ Done! Captions saved to {OUTPUT_JSON}")


# plt.tight_layout()
# plt.show()
