https://github.com/CompVis/latent-diffusion.git

```
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

```
pip install einops fire gradio==5.17.1 huggingface-hub invisible-watermark matplotlib numpy opencv-python Pillow requests safetensors scikit-learn scipy scikit-image tqdm transformers==4.49.0 sentencepiece
```

```
python cli_invert_target.py \
  --target_image blue_dog.png \
  --mask ref_mask1111.jpg 
```



```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
```
!pip install timm diffusers transformers accelerate pandas numpy Pillow tqdm opencv-python-headless
```
```
!pip install controlnet_aux open_clip_torch xformers
```


```
python cli_kv_edit.py \
    --input_image "blue_dog.png" \
    --mask_image "ref_mask1111.jpg" \
    --source_prompt "a cat is lying on the floor" \
    --target_prompt "a dog is lying on the floor"

```
