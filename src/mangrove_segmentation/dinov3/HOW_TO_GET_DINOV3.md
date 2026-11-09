# How to get DINOv3 and run it on your PC

DINOv3 is Meta's image foundation model. It is free to use, but the weights are **gated**: you must accept Meta's
DINOv3 licence and log in to Hugging Face once. After that, the code downloads the model by itself.

## 1. Accept the licence (once)

1. Create a free account at https://huggingface.co/join and confirm your email.
2. Open the model page and click the button to **agree and access the repository** (you fill in a short form):
   - recommended, satellite-trained: https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m
   - small and fast (if you have no GPU or little RAM): https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
3. Approval is often quick but **can take a while**. The model page says when you have access.
   Accept both if you may want to compare them.

## 2. Log in on your PC (once)

1. On Hugging Face: your picture (top right) → **Settings → Access Tokens → Create new token**. Type **Read**, any name.
2. Open the **Anaconda Prompt**, then:
   ```
   conda activate mangrove
   pip install -r %USERPROFILE%\Desktop\SDP\2_Code\dinov3\requirements_dinov3.txt
   huggingface-cli login
   ```
3. Paste the token **into the Anaconda Prompt** when it asks (nothing is shown while you paste, that's normal), press
   Enter, and answer `n` to the git-credential question.
   Never paste the token into a notebook, a script or a chat. Alternatively, set it yourself as the Windows
   environment variable `HF_TOKEN`.
4. Check: `python -c "from huggingface_hub import whoami; print(whoami()['name'])"` prints your user name.

## 3. Which model?

| Model | What it learned from | Download | RAM needed | Best for |
| --- | --- | --- | --- | --- |
| `facebook/dinov3-vitl16-pretrain-sat493m` (ViT-L/16, 300M parameters) **(default)** | 493 million Maxar satellite images (RGB, 0.6 m) | about 1.2 GB | 8 GB+ | the real experiment |
| `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S/16, 21M parameters) | web photos | about 90 MB | 4 GB+ | a quick first run, weak PCs |

Our chips are 10 m Sentinel-2, about 17 times coarser than what the satellite model saw. Finding out whether DINOv3
still helps at that resolution is part of the research question.

## 4. Run it

From the SDP folder in the Anaconda Prompt (`cd %USERPROFILE%\Desktop\SDP`, environment active):

| Command | What it does | Expected time |
| --- | --- | --- |
| `python 2_Code\dinov3\train_dinov3.py --mode frozen` | Stage A, all 5 folds, satellite ViT-L | feature extraction once: about 5–15 min on a CPU for the 215 chips (seconds on a GPU); then about 5–15 min per fold on a CPU |
| `python 2_Code\dinov3\train_dinov3.py --mode frozen --no-spectral` | ablation: DINOv3 RGB only | reuses the cached features, so training time only |
| `python 2_Code\dinov3\train_dinov3.py --model-id facebook/dinov3-vits16-pretrain-lvd1689m` | small model | a few minutes of extraction |
| `python 2_Code\dinov3\train_dinov3.py --mode lora` | Stage B, LoRA r=8 on attention q/v | **NVIDIA GPU only**: about 8 GB of GPU memory for ViT-L (use `--batch 2` if it runs out); tens of minutes per fold |

Or use the notebook `2_Code\notebooks\04_dinov3.ipynb`, which has the same choices as switches.

Notes
- The data must exist first: `python 2_Code\run_all.py --step download`, then `--step prepare` and `--step datasets`
  (or notebook 00). DINOv3 reads only `6_Work\datasets\Train_Val_Dataset`. The test set stays locked.
- Features are cached in `6_Work\dinov3_cache` (about 450 MB for ViT-L, 170 MB for ViT-S). Delete that folder to
  re-extract them.
- Results: `4_Results\DINOv3\<run name>\RESULTS.md`, with the comparison against B1, B2 (0.704) and the U-Net, per-fold
  scores, figures and the saved decoders.
- To test the code without any weights or login: `--smoke --out %TEMP%\dino_smoke` (a tiny random model; the numbers mean nothing).

## Troubleshooting

- **"Could not load … gated / 401 / 403":** the licence isn't accepted yet, or you aren't logged in (step 1 or 2).
- **"Unrecognized model type dinov3_vit":** transformers is too old. Run `pip install -U "transformers>=4.56"`.
- **Out of memory on CPU:** use the ViT-S model, or close other programs.
- **Out of memory on GPU (LoRA):** `--batch 2`, or `--model-id facebook/dinov3-vits16-pretrain-lvd1689m`.
- **Slow on CPU:** frozen mode is the CPU option. LoRA refuses to run on a CPU (it would take days).
