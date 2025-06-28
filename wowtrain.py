import random
import cv2
import einops
import numpy as np
import torch
import pandas as pd
from PIL import Image
from pytorch_lightning import seed_everything

from data import HWC3, apply_color, resize_image
from ddim import DDIMSampler
from model import create_model, load_state_dict

# 모델 로드
model = create_model('cldm_v21.yaml').cpu() #./models/
model.load_state_dict(load_state_dict(
    'colorizenet-sd21.ckpt', location='cuda')) #lightning_logs/version_6/checkpoints/
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# 하이퍼파라미터
num_samples = 1
guess_mode = False
strength = 1.0
eta = 0.0
ddim_steps = 20
scale = 9.0
seed = 1294574436
seed_everything(seed)

# 결과 저장 리스트
out_imgs = []
out_img_names = []

# CSV 읽기
test_df = pd.read_csv("test.csv")

# 반복 수행
for img_id, img_path, caption in zip(test_df['ID'], test_df['input_img_path'], test_df['caption']):
    # 이미지 읽고 전처리
    input_image = cv2.imread(img_path)
    input_image = HWC3(input_image)
    img = resize_image(input_image, resolution=512)
    H, W, C = img.shape

    control = torch.from_numpy(img.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    full_prompt = f"realistic, high quality, detailed, Do not change the structure. Only Colorize. {caption}"

    # conditioning 구성
    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([full_prompt] * num_samples)]
    }
    un_cond = {
        "c_concat": None if guess_mode else [control],
        "c_crossattn": [model.get_learned_conditioning([""] * num_samples)]
    }
    shape = (4, H // 8, W // 8)
    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

    # 샘플링
    samples, _ = ddim_sampler.sample(ddim_steps, num_samples, shape, cond,
                                     verbose=False, eta=eta,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=un_cond)

    # 후처리
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    colored_results = [apply_color(img, result) for result in results]

    out_imgs.append(colored_results[0])  # 단일 샘플이므로 첫 번째만 사용
    out_img_names.append(img_id)

print("✅ Test 데이터셋에 대한 모든 이미지 생성 완료.")






#추론 submission만들기
os.makedirs(CFG['SUB_DIR'], exist_ok=True)

# **중요** 추론 이미지 평가용 Embedding 추출 모델
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai") # 모델명을 반드시 일치시켜야합니다.

clip_model.to(device)
# 평가 제출을 위해 추론된 이미지들을 ViT-L-14 모델로 임베딩 벡터(Feature)를 추출합니다.
feat_imgs = []
for output_img, img_id in tqdm(zip(out_imgs, out_img_names)):
    path_out_img = CFG['SUB_DIR'] + '/' + img_id + '.png'
    output_img.save(path_out_img)
    # 평가용 임베딩 생성 및 저장
    output_img = clip_preprocess(output_img).unsqueeze(0).cuda()
    with torch.no_grad():
        feat_img = clip_model.encode_image(output_img)
        feat_img /= feat_img.norm(dim=-1, keepdim=True) # L2 정규화 필수

    feat_img = feat_img.detach().cpu().numpy().reshape(-1)
    feat_imgs.append(feat_img)

feat_imgs = np.array(feat_imgs)
vec_columns = [f'vec_{i}' for i in range(feat_imgs.shape[1])]
feat_submission = pd.DataFrame(feat_imgs, columns=vec_columns)
feat_submission.insert(0, 'ID', out_img_names)
feat_submission.to_csv(CFG['SUB_DIR']+'/embed_submission.csv', index=False)

# 최종 제출물 (ZIP) 생성 경로
# 제출물 (ZIP) 내에는 디렉토리(폴더)가 없이 구성해야합니다.
zip_path = './submission.zip'

# zip 파일 생성
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_name in os.listdir(CFG['SUB_DIR']):
        file_path = os.path.join(CFG['SUB_DIR'], file_name)

        # 일반 파일이며 숨김 파일이 아닌 경우만 포함
        if os.path.isfile(file_path) and not file_name.startswith('.'):
            zipf.write(file_path, arcname=file_name)

print(f"✅ 압축 완료: {zip_path}")
