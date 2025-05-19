import os, math, torch, numpy as np, torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
from typing import List
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoConfig

# --------------------------------------------------------------------------
# ↓ helpers copied from official InternVL3 demo (unchanged) ↓
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _find_closest(r, targets, W, H, sz):
    best, diff = (1,1), float("inf")
    area = W * H
    for x,y in targets:
        d = abs(r - x/y)
        if d < diff or (d == diff and area > 0.5*sz*sz*x*y):
            best, diff = (x,y), d
    return best

def dynamic_preprocess(img, *, min_num=1, max_num=12, image_size=448,
                       use_thumbnail=True):
    W, H = img.size
    r    = W/H
    targets = sorted({(i,j) for n in range(min_num,max_num+1)
                      for i in range(1,n+1) for j in range(1,n+1)
                      if 1 <= i*j <= max_num},
                     key=lambda v: v[0]*v[1])
    gx, gy = _find_closest(r, targets, W, H, image_size)
    w, h   = image_size*gx, image_size*gy
    blocks = gx*gy
    img_rs = img.resize((w,h))
    tiles  = [img_rs.crop(((i%gx)*image_size,
                           (i//gx)*image_size,
                           (i%gx+1)*image_size,
                           (i//gx+1)*image_size))
              for i in range(blocks)]
    if use_thumbnail and blocks != 1:
        tiles.append(img.resize((image_size, image_size)))
    return tiles

def load_image(path, *, input_size=448, max_num=12):
    tfm   = build_transform(input_size)
    tiles = dynamic_preprocess(Image.open(path).convert("RGB"),
                               image_size=input_size, max_num=max_num)
    return torch.stack([tfm(t) for t in tiles])

def split_model(model_name):
    world = torch.cuda.device_count()
    conf  = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    L     = conf.llm_config.num_hidden_layers
    per   = math.ceil(L / (world-0.5))
    per   = [per]*world; per[0] = math.ceil(per[0]*0.5)
    mp, cnt = {}, 0
    for i,n in enumerate(per):
        for _ in range(n):
            mp[f"language_model.model.layers.{cnt}"] = i
            cnt += 1
    mp.update({
        "vision_model": 0, "mlp1": 0,
        "language_model.model.tok_embeddings": 0,
        "language_model.model.embed_tokens":   0,
        "language_model.output":               0,
        "language_model.model.norm":           0,
        "language_model.model.rotary_emb":     0,
        "language_model.lm_head":              0,
        f"language_model.model.layers.{L-1}":  0,
    })
    return mp
# --------------------------------------------------------------------------

# ========= USER CONFIG (unchanged) =========
MODEL_PATH   = "OpenGVLab/InternVL3-14B"
IMAGE_FOLDER = "All_Images_formatted_final_dataset"
CSV_PATH     = "Corrected_Questions - FINAL_DATASET_ENGLISH.csv"
QUESTION_COLUMNS = {
    "Refactored question":            "r",
    "Common Sense Cultural Question": "cscq",
    "Multi-hop reasoning Question":   "mhr",
    "Analogy Question":               "an",
}
GEN_CFG = dict(max_new_tokens=64, do_sample=False, temperature=0.0)

# ========= load model/tokenizer =========
device_map = split_model("InternVL3-14B")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
model      = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,         # switch flags as you prefer
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map
).eval()

# ========= original prompts (unchanged) =========
def make_standard_prompt(q, o):
    return (
        "<image>\n"
        f"Question: {q}\n"
        "Options:\n"
        f"1. {o[0]}\n2. {o[1]}\n3. {o[2]}\n4. {o[3]}\n\n"
        "Based on the image and question, select and output the correct option. "
        "Do not output any other explanation."
    )

CULTURAL_PROMPT = (
    "<image>\n"
    "You are an expert analyst deeply knowledgeable in Indian culture, traditions, and regional heritage. "
    "Carefully analyze the provided image and question. Reason methodically through each of the following culturally informed dimensions to identify the correct answer. "
    "Please output only the correct option/answer from the given options without any additional information or reasoning steps.\n\n"
    "Dimension A – Drishti (Visual Insight)\n"
    "•  Carefully examine the image, identifying culturally significant visual elements such as attire, architecture, rituals, landscapes, or symbols.\n\n"
    "Dimension B – Smriti (Cultural Memory)\n"
    "•  Recall relevant historical details, traditional knowledge, or well-known cultural practices from India related to this question.\n\n"
    "Dimension C – Yukti (Logical Integration)\n"
    "•  Logically integrate your observations from Drishti and knowledge from Smriti. Use this integration to rule out options that are culturally or logically inconsistent.\n\n"
    "Dimension D – Sthiti (Regional Contextualization)\n"
    "•  Consider regional and cultural contexts within India. Determine which provided option best aligns with the cultural and regional insights you've gained.\n\n"
    "Question: {q}\n"
    "Options:\n1. {o1}\n2. {o2}\n3. {o3}\n4. {o4}\n\n"
    "Output Format:\nAfter evaluating all cultural perspectives, respond with only the correct option (e.g. “1” or “Option1”) and do not include any reasoning steps, chain-of-thought, or additional explanation."
)

# ========= inference wrappers =========
def _run_chat(pixel, prompt):
    return model.chat(tokenizer, pixel, prompt, GEN_CFG)

def infer_standard(q, opts, img):
    pv  = load_image(img, max_num=12).to(torch.bfloat16).cuda()
    raw = _run_chat(pv, make_standard_prompt(q, opts))
    return opts[int(raw[0])-1] if raw and raw[0] in "1234" else raw

def infer_cultural(q, opts, img):
    pv  = load_image(img, max_num=12).to(torch.bfloat16).cuda()
    prompt = CULTURAL_PROMPT.format(q=q, o1=opts[0], o2=opts[1], o3=opts[2], o4=opts[3])
    raw = _run_chat(pv, prompt)
    return opts[int(raw[0])-1] if raw and raw[0] in "1234" else raw

# ========= CSV logic (unchanged) =========
def process_column(qcol, prefix, df, infer_fn):
    out_csv = f"{prefix}_output_with_answers_intern.csv"
    ckpt    = f"checkpoint_{prefix}_intern.pkl"
    df_out  = df.copy()
    df_out["ChosenOption"] = ""
    start = 0
    if os.path.exists(ckpt):
        df_out = pd.read_pickle(ckpt)
        m = df_out["ChosenOption"].eq("")
        if m.any(): start = df_out[m].index[0]

    idxs = list(df_out.index[start:])
    for idx in tqdm(idxs, total=len(idxs), desc=f"InternVL {prefix}"):
        row   = df_out.loc[idx]
        q     = row[qcol]
        opts  = [row[f"Option{i}"] for i in range(1,5)]
        img   = row["Image Name"]
        path  = img if img.startswith("http") else os.path.join(IMAGE_FOLDER, img)

        if not q or pd.isna(q):
            ans = "Error: Missing Question"
        elif not path.startswith("http") and not os.path.isfile(path):
            ans = f"Error: Missing Image '{img}'"
        else:
            try: ans = infer_fn(q, opts, path)
            except AssertionError as e: ans = f"Error: AssertionError ({e})"
            except Exception      as e: ans = f"Error: {type(e).__name__}: {e}"

        df_out.at[idx, "ChosenOption"] = ans
        cols = [qcol, *[f"Option{i}" for i in range(1,5)], "Image Name", "ChosenOption"]
        df_out.loc[[idx], cols].to_csv(out_csv,
                                       mode="w" if idx == idxs[0] else "a",
                                       header=idx == idxs[0], index=False)
        if (idxs.index(idx)+1) % 500 == 0:
            df_out.to_pickle(ckpt)
    df_out.to_pickle(ckpt)

# ========= main =========
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(f"{IMAGE_FOLDER} not found")

    for qcol, prefix in QUESTION_COLUMNS.items():
        need = [qcol, *[f"Option{i}" for i in range(1,5)], "Image Name"]
        if any(c not in df.columns for c in need):
            print(f"Skipping {prefix} (missing columns)"); continue

        process_column(qcol, prefix, df, infer_standard)
        if prefix == "r":
            process_column(qcol, "cc", df, infer_cultural)

    print("\nGenerated CSV files:")
    for p in list(QUESTION_COLUMNS.values()) + ["cc"]:
        print(f"- {p}_output_with_answers_intern.csv")
