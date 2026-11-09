"""Builds 2_Code/notebooks/04_dinov3.ipynb (clean, no outputs). Run: python 2_Code/dinov3/make_notebook_04.py"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 04 · DINOv3: a frozen foundation model + light tuning for mangrove mapping\n"
       "This notebook runs the **DINOv3 route** from the proposal and shows its results next to the baselines and the U-Net.\n\n"
       "* **Stage A, `frozen` (works on a normal PC):** DINOv3 looks at the RGB of every chip once, its features are saved "
       "to disk, and only a small decoder is trained. The decoder also sees all 8 spectral bands (near-infrared, SWIR, NDVI...), "
       "which DINOv3 itself can't see.\n"
       "* **Stage B, `lora` (needs an NVIDIA GPU):** small LoRA adapters inside DINOv3 are trained too.\n\n"
       "Same rules as before: Train/Val data only, 5 site folds, the test set stays locked. The bar to beat: "
       "**random forest IoU 0.704**."),
    md("## Before you start (once)\n"
       "DINOv3's weights are free, but you must accept Meta's licence and log in. Follow "
       "`2_Code/dinov3/HOW_TO_GET_DINOV3.md`. In short: make a Hugging Face account, accept the licence on the model page, "
       "then run `huggingface-cli login` in the **Anaconda Prompt** and paste your token there (never in this notebook)."),
    code("import sys\nfrom pathlib import Path\n\n"
         "CODE = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd() / '2_Code'\n"
         "for p in (CODE, CODE / 'dinov3'):\n"
         "    if str(p) not in sys.path:\n"
         "        sys.path.insert(0, str(p))\n"
         "import paths as P\nimport dino_paths as DP\nimport train_dinov3 as T\n"
         "print(P.describe())\nprint('DINOv3 results:', DP.DINO_RESULTS)\nprint('Train/Val data :', DP.TRAIN_VAL, '(exists)' if DP.TRAIN_VAL.exists() else '(MISSING: run 00_get_data first)')"),
    md("## Choose the model and the run"),
    code("MODEL_ID    = 'facebook/dinov3-vitl16-pretrain-sat493m'   # satellite-trained ViT-L (recommended, ~1.2 GB)\n"
         "# MODEL_ID  = 'facebook/dinov3-vits16-pretrain-lvd1689m'  # small ViT-S: if you have no GPU or little RAM\n"
         "MODE        = 'frozen'   # 'frozen' (any PC) or 'lora' (NVIDIA GPU only)\n"
         "NO_SPECTRAL = False      # True = ablation: DINOv3 RGB features only\n"
         "FOLDS       = None       # None = all 5 folds, or e.g. [1, 2]\n"
         "EPOCHS      = None       # None = default (frozen 40, lora 30)\n"
         "RUN_TRAINING = False     # set True to train (frozen on a CPU: feature extraction plus about 5-15 min per fold)\n"
         "SMOKE       = False      # True = tiny random model, tests the code without weights (results are meaningless)\n"
         "MAX_CHIPS   = None       # e.g. 4 for a very quick test"),
    md("## Check the Hugging Face login and licence (skipped in smoke mode)"),
    code("if not SMOKE:\n"
         "    try:\n"
         "        from huggingface_hub import whoami, hf_hub_download\n"
         "        print('logged in as:', whoami()['name'])\n"
         "        hf_hub_download(MODEL_ID, 'config.json')\n"
         "        print('licence accepted: you can download', MODEL_ID)\n"
         "    except Exception as e:\n"
         "        print('NOT READY:', type(e).__name__, str(e)[:200])\n"
         "        print('-> follow 2_Code/dinov3/HOW_TO_GET_DINOV3.md (accept the licence, then huggingface-cli login)')\n"
         "else:\n"
         "    print('smoke mode: no login needed')"),
    md("## Train (only if `RUN_TRAINING = True`)\n"
       "Progress is printed per epoch. Results go to `4_Results/DINOv3/<run name>/` (smoke runs go to a temporary folder)."),
    code("import tempfile\nRUN_DIR = None\n"
         "if RUN_TRAINING:\n"
         "    argv = ['--model-id', MODEL_ID, '--mode', MODE]\n"
         "    if NO_SPECTRAL: argv.append('--no-spectral')\n"
         "    if FOLDS: argv += ['--folds', *map(str, FOLDS)]\n"
         "    if EPOCHS: argv += ['--epochs', str(EPOCHS)]\n"
         "    if MAX_CHIPS: argv += ['--max-chips', str(MAX_CHIPS)]\n"
         "    if SMOKE:\n"
         "        tmp = Path(tempfile.mkdtemp(prefix='dinov3_smoke_'))\n"
         "        argv += ['--smoke', '--out', str(tmp), '--cache-dir', str(tmp / 'cache')]\n"
         "        if MODE == 'lora': argv.append('--force')\n"
         "    RUN_DIR = T.run(T.parse(argv))\n"
         "else:\n"
         "    print('RUN_TRAINING is False: showing existing results only')"),
    md("## Results"),
    code("from IPython.display import Markdown, Image, display\n"
         "runs = [RUN_DIR] if RUN_DIR else sorted([d for d in DP.DINO_RESULTS.glob('*') if (d / 'RESULTS.md').exists()],\n"
         "                                       key=lambda d: (d / 'RESULTS.md').stat().st_mtime)\n"
         "if not runs:\n"
         "    print('No DINOv3 results yet. Set RUN_TRAINING = True above and run the notebook again.')\n"
         "else:\n"
         "    print('runs found:', [r.name for r in runs])\n"
         "    run = runs[-1]\n"
         "    display(Markdown((run / 'RESULTS.md').read_text(encoding='utf-8')))\n"
         "    for f in ('iou_per_fold.png', 'training_curves.png', 'examples.png'):\n"
         "        if (run / 'figs' / f).exists():\n"
         "            display(Image(filename=str(run / 'figs' / f)))"),
    md("## Next\n"
       "* **Beat 0.704?** If frozen DINOv3 is close, try `MODE = 'lora'` on a GPU (Colab Pro or an NVIDIA PC).\n"
       "* **Is DINOv3 helping at all?** Run once with `NO_SPECTRAL = True`. If RGB-only DINOv3 is far below the "
       "full version, most of the skill comes from the spectral bands, not from DINOv3 (the same lesson as the old SAM work).\n"
       "* The test set is scored **once**, only for the final chosen model."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).resolve().parent.parent / "notebooks" / "04_dinov3.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
