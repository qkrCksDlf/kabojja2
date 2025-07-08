from tqdm import tqdm
import random
import os
import zipfile
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
import open_clip
import re
import warnings
warnings.filterwarnings('ignore')

# 1. 기본 설정 및 시드 고정

CFG = {
    'DRIVE_PATH': '/content/drive/MyDrive/INHACHALLANGE/Data', # ⚠️ 본인 데이터 경로에 맞게 수정
    'SUB_DIR': './submission',
    'ENSEMBLE_SEEDS': [42, 123, 1024] # 앙상블에 사용할 시드 리스트
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 2. 모델 통합 로딩 (최적화)

print("Loading all models...")

# [ControlNet] Canny와 Depth 모델 로드 (다중 ControlNet)
controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch_dtype
)
controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth", torch_dtype=torch_dtype
)
controlnets = [controlnet_canny, controlnet_depth]

# [Stable Diffusion] 메인 파이프라인 로드
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnets,
    torch_dtype=torch_dtype,
    safety_checker=None # 안전 검사기 비활성화 (선택 사항)
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# xformers를 사용한 메모리 최적화 (설치된 경우)
try:
    pipe.enable_xformers_memory_efficient_attention()
except ImportError:
    print("xformers not installed. For better performance, install it with 'pip install xformers'")


# [ControlNet Preprocessor] Canny, Depth 감지기 로드
from controlnet_aux import CannyDetector
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Canny 감지기
canny_detector = CannyDetector()

# DepthAnything 직접 로드
depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-base-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-base-hf").to(device)
depth_model.eval()

# Depth 추론 함수
def get_depth_map(image_pil):
    inputs = depth_processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
    depth = outputs.predicted_depth.squeeze().cpu().numpy()
    # 0-1 정규화
    depth_min, depth_max = depth.min(), depth.max()
    normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    return normalized


# [BLIP-2] 프롬프트 생성을 위한 BLIP-2 모델 로드
from transformers import AutoProcessor, Blip2ForConditionalGeneration
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", device_map="auto", torch_dtype=torch_dtype
)

# [OpenCLIP] 최종 제출용 임베딩을 위한 CLIP 모델 로드
import open_clip
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
clip_model.to(device).eval()

print("✅ All models loaded successfully!")

# 3. 헬퍼 함수 정의

def generate_blip2_caption(image: Image.Image) -> str:
    """BLIP-2를 사용하여 이미지로부터 상세 캡션을 생성합니다."""
    inputs = blip_processor(images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=20)
    generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def get_control_images(image: Image.Image) -> list[Image.Image]:
    """입력 이미지로부터 Canny, Depth 제어 이미지를 생성합니다."""
    image_np = np.array(image)

    # Canny 이미지 생성
    canny_image = Image.fromarray(canny_detector(image_np))

    # Depth 이미지 생성
    depth_map = get_depth_map(image)                # numpy 2D array
    depth_image = depth_map_to_pil(depth_map)       # PIL 이미지

    return [canny_image, depth_image]

def depth_map_to_pil(depth_map):
    depth_uint8 = (depth_map * 255).astype(np.uint8)
    return Image.fromarray(depth_uint8)

def preprocess_caption(caption: str) -> str:
    # 1. 질문 형식의 문장 제거
    # '?'로 끝나거나 특정 질문 단어로 시작하는 문장 제거
    sentences = re.split(r'(?<=[.!?])\s+', caption.strip()) # 마침표, 물음표, 느낌표 기준으로 문장 분리

    filtered_sentences = []
    for s in sentences:
        s_lower = s.lower()
        if s.endswith('?') or \
           s_lower.startswith('what is') or \
           s_lower.startswith('do you see') or \
           s_lower.startswith('are there') or \
           s_lower.startswith('which piece') or \
           s_lower.startswith('is the'):
            continue # 질문 문장은 건너뛰기

        # 2. 불필요한 시작 구문 제거 (단, 문장 시작 부분에만 적용)
        if s_lower.startswith('in this image i can see'):
            s = s[len('in this image i can see'):].strip()
            # 첫 글자가 소문자면 대문자로 변경 (문법적으로)
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
        elif s_lower.startswith('i can see'): # 다른 유사한 불필요 구문이 있다면 추가
            s = s[len('i can see'):].strip()
            if s and s[0].islower():
                s = s[0].upper() + s[1:]

        filtered_sentences.append(s.strip())

    return " ".join(filtered_sentences).strip()

# 4. 메인 로직 (앙상블 적용)

# 데이터 로드
test_df = pd.read_csv(os.path.join(CFG['DRIVE_PATH'], 'test.csv'))
test_df['input_img_path'] = test_df['ID'].apply(
    lambda x: os.path.join(CFG['DRIVE_PATH'], f'test/input_image/{x}.png')
)

all_seed_embeddings = [] # 각 시드별 결과 임베딩을 저장할 리스트

# [앙상블 루프]
for seed in CFG['ENSEMBLE_SEEDS']:
    print(f"\n--- Running generation for seed: {seed} ---")
    seed_everything(seed)

    out_imgs = []
    out_img_names = []

    # [이미지 생성 루프]
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Seed {seed}"):
        img_id, img_path, original_caption = row['ID'], row['input_img_path'], row['caption']

        original_caption = preprocess_caption(original_caption)
        # 1. 이미지 불러오기
        input_img = Image.open(img_path).convert("RGB")

        # 2. 제어 이미지 생성 (Canny, Depth)
        control_images = get_control_images(input_img)

        # 3. BLIP-2로 프롬프트 고도화
        # blip_caption = generate_blip2_caption(input_img)
        final_prompt = f"high-res. {original_caption}"
        negative_prompt = "low quality, worst quality, ugly, deformed, blurry"

        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        tokens = tokenizer.encode(final_prompt, truncation=False) # truncation=False로 설정하여 자르지 않고 모든 토큰을 확인
        # print(f"ID: {img_id}, Blip result: {blip_caption}")
        print(f"ID: {img_id}, Original Prompt: {original_caption}")
        print(f"ID: {img_id}, Final Prompt Token Length: {len(tokens)}")
        print(f"ID: {img_id}, Truncated?: {len(tokens) > 77}")

        # 4. 이미지 생성 (다중 ControlNet 적용)
        result = pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=control_images, # 제어 이미지를 리스트로 전달
            guidance_scale=6.8,
            num_inference_steps=40,
            controlnet_conditioning_scale=[0.5, 0.8] # [canny_scale, depth_scale]
        ).images[0]

        out_imgs.append(result)
        out_img_names.append(img_id)

    # 5. 현재 시드의 결과로부터 CLIP 임베딩 추출
    seed_feats = []
    for img in tqdm(out_imgs, desc="Extracting CLIP embeddings"):
        image_tensor = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(image_tensor)
            feat /= feat.norm(dim=-1, keepdim=True) # L2 정규화
            seed_feats.append(feat.cpu().numpy())

    all_seed_embeddings.append(np.vstack(seed_feats))
    print(f"✅ Finished run for seed: {seed}")
# 5. 최종 결과 집계 및 제출 파일 생성

print("\n--- Averaging embeddings from all seeds for final submission ---")
# [앙상블] 모든 시드의 임베딩을 평균
final_embeddings = np.mean(all_seed_embeddings, axis=0)

# 제출 디렉토리 생성
os.makedirs(CFG['SUB_DIR'], exist_ok=True)

# 마지막 시드에서 생성된 이미지 저장 (시각적 확인용)
for img, img_id in zip(out_imgs, out_img_names):
    img.save(os.path.join(CFG['SUB_DIR'], f"{img_id}.png"))

# 평균 임베딩을 CSV로 저장
vec_columns = [f'vec_{i}' for i in range(final_embeddings.shape[1])]
feat_submission = pd.DataFrame(final_embeddings, columns=vec_columns)
feat_submission.insert(0, 'ID', out_img_names)
feat_submission.to_csv(os.path.join(CFG['SUB_DIR'], 'embed_submission.csv'), index=False)
print("Final embedding CSV saved.")

# ZIP 압축
zip_path = './submission.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_name in os.listdir(CFG['SUB_DIR']):
        file_path = os.path.join(CFG['SUB_DIR'], file_name)
        zipf.write(file_path, arcname=file_name)

print(f"Submission file '{zip_path}' created successfully!")

