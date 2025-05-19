import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Directory and model folders
BASE_DIR = "/home/nemilai/Desktop/IITP/Outputs"
OUTPUT_DIRS = [
    "outputs_gemma",
    "outputs_janus",
    "outputs_chitrarth",
    "outputs_gpt4o_mini_all",
    "outputs_intern_llm",
    "outputs_intern_slm",
    "outputs_kimivl",
    "outputs_llama_llm",
    "outputs_llava_llm",
    "outputs_maya",
    "outputs_qwenomni",
    "outputs_qwenvl",
    "outputs_smol",
]

# Load accuracy per language for each model
def load_language_accuracy():
    model_lang_acc = {}
    languages = set()
    for out_dir in OUTPUT_DIRS:
        dir_path = os.path.join(BASE_DIR, out_dir)
        if not os.path.exists(dir_path):
            continue
        model = out_dir.replace('outputs_', '')
        model_lang_acc[model] = {}
        for csv_file in glob.glob(os.path.join(dir_path, '*.csv')):
            name = os.path.basename(csv_file)
            lang = name.split('_')[0]  # assumes filename like 'assamese_model.csv'
            df = pd.read_csv(csv_file)
            if 'predicted_correctly' not in df.columns:
                continue
            acc = df['predicted_correctly'].mean()
            model_lang_acc[model][lang] = acc
            languages.add(lang)
    return model_lang_acc, sorted(languages)

# Create 3D scatter
def create_3d_scatter(model_lang_acc, languages):
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(bottom=0.35, top=0.9)
    ax = fig.add_subplot(111, projection='3d')
    models = list(model_lang_acc.keys())
    cmap = plt.get_cmap('tab10')

    for idx, model in enumerate(models):
        langs = list(model_lang_acc[model].keys())
        xs = [languages.index(lang) for lang in langs]
        ys = [model_lang_acc[model][lang] for lang in langs]
        zs = [idx] * len(xs)
        ax.scatter(xs, ys, zs, color=cmap(idx % 10), label=model, s=50)

    # Axes labels
    ax.set_xticks(range(len(languages)))
    ax.set_xticklabels(languages, rotation=90, ha='center', fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_zlim(-0.5, len(models) - 0.5)
    ax.set_zticks(range(len(models)))
    ax.set_zticklabels(models)
    ax.set_xlabel('Language', labelpad=20)
    ax.set_zlabel('Model', labelpad=20)

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    plt.tight_layout()
    # Save
    plt.savefig(os.path.join(BASE_DIR, '3d_scatter_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, '3d_scatter_accuracy.pdf'), bbox_inches='tight')
    plt.show()

# 2D accuracy plots by question type and model

def load_qtype_accuracy():
    all_dfs = []
    for out_dir in OUTPUT_DIRS:
        dir_path = os.path.join(BASE_DIR, out_dir)
        if not os.path.exists(dir_path):
            continue
        for csv_file in glob.glob(os.path.join(dir_path, '*.csv')):
            lang = os.path.basename(csv_file).split('_')[0]
            df = pd.read_csv(csv_file)
            if 'predicted_correctly' not in df.columns or 'question_type' not in df.columns:
                continue
            tmp = df[['question_type', 'predicted_correctly']].copy()
            tmp['language'] = lang
            all_dfs.append(tmp)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def create_qtype_bar_chart(df):
    pivot = df.groupby(['language', 'question_type'])['predicted_correctly']\
              .mean().unstack(fill_value=0)
    pivot = pivot.sort_index()
    ax = pivot.plot(kind='bar', figsize=(12,6), rot=45)
    ax.set_xlabel('Language')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Language and Question Type')
    ax.legend(title='Question Type', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'accuracy_by_qtype.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'accuracy_by_qtype.pdf'), bbox_inches='tight')
    plt.show()

def create_model_bar_chart(model_lang_acc):
    models = list(model_lang_acc.keys())
    accs = [np.mean(list(model_lang_acc[m].values())) for m in models]
    plt.figure(figsize=(12,6))
    plt.bar(models, accs, color=plt.get_cmap('tab10').colors[:len(models)])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Mean Accuracy per Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'accuracy_by_model.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(BASE_DIR, 'accuracy_by_model.pdf'), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    data, langs = load_language_accuracy()
    create_3d_scatter(data, langs)
    # 2D plots
    combined_df = load_qtype_accuracy()
    if not combined_df.empty:
        create_qtype_bar_chart(combined_df)
    create_model_bar_chart(data)
