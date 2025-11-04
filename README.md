## DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models' Understanding on Indian Culture

### Overview

DRISHTIKON is a **first-of-its-kind multimodal, multilingual benchmark** dedicated to evaluating Vision-Language Models’ (VLMs) ability to understand Indian culture. Unlike existing global benchmarks, DRISHTIKON focuses exclusively on India’s cultural richness, spanning **15 languages, all 28 states, and 8 union territories**, with **64,288 carefully curated text-image pairs**.

The dataset captures **festivals, attire, cuisines, rituals, art forms, architecture, personalities, and heritage**, enabling **fine-grained evaluation of cultural reasoning** in multimodal models.

* **Paper:** [DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models' Understanding on Indian Culture](https://huggingface.co/papers/2509.19274)
* **Dataset:** [Hugging Face](https://huggingface.co/datasets/13ari/DRISHTIKON)

---

## 🎯 Introduction and Goal

Existing VLM benchmarks often suffer from **"cultural blindness,"** failing to understand crucial socio-cultural contexts, particularly in diverse regions like India. DRISHTIKON addresses this by offering a culturally specific evaluation that requires inferential chaining and visual grounding in the context of Indian heritage.

The core goal is to test a model's ability to reason over multimodal inputs (image-text pairs) that are deeply rooted in Indian culture.

## ✨ Key Dataset Features

  * **Size:** Over 64,000 (specifically **64,288**) meticulously curated image-text pairs.
  * **Geographic Coverage:** Spans all **28 states and 8 Union Territories** of India.
  * **Multilingual:** Covers **15 diverse languages** (14 Indic languages plus English). The Indic languages include Hindi, Punjabi, Odia, Gujarati, Assamese, Malayalam, Urdu, Tamil, Kannada, Telugu, Konkani, Bengali, Sindhi, and Marathi.
  * **Attributes:** Captures **16 fine-grained attributes** of Indian culture, including:
      * Rituals and Ceremonies 
      * History
      * Tourism
      * Cuisine 
      * Dance and Music 
      * Art 
      * Festivals
      * Religion 
      * *...and more* (Costume, Medicine, Nightlife, Personalities, Language, Sports, Transport, Cultural Common Sense).

-----

## 📊 Dataset Statistics & Distribution

<img width="569" height="586" alt="image" src="https://github.com/user-attachments/assets/1c3bd7a4-68dc-40ff-91b8-ac9d219c1654" />

<img width="1316" height="667" alt="image" src="https://github.com/user-attachments/assets/f0fee265-f640-4ee9-9828-90e892540046" />


The dataset provides comprehensive coverage across regions and cultural themes.

The most frequent attributes based on the bar chart are:

  * Cultural Common Sense (**14,085** questions) 
  * History (**11,055** questions) 
  * Rituals and Ceremonies (**7,005** questions) 

-----

## 🧠 Question Categories and Reasoning

The benchmark features multiple-choice questions (MCQs) across four main categories.

| Question Category | Count (Original English) | Description |
| :--- | :--- | :--- |
| **General Question** | 2,126 questions | Simple factual questions |
| **Analogy Question** | 720 questions | Requires inferring the answer by relating cultural equivalents or symbols |
| **Multi-hop Reasoning** | 720 questions | Requires connecting at least two facts (visual/cultural/historical) to reach the answer |
| **Common Sense Cultural** | 720 questions | Requires engaging with culturally grounded knowledge that is not explicitly stated |

### Culturally Grounded Chain-of-Thought (CoT)

<img width="1090" height="686" alt="image" src="https://github.com/user-attachments/assets/36cf85ac-981b-4a39-95b9-e3326cc09755" />

For Chain-of-Thought (CoT) evaluation, a unique, culturally informed process is used, drawing from classical Indian epistemology:

1.  **Drishti (Visual Insight):** Examine visual elements (attire, architecture, symbols).
2.  **Smriti (Cultural Memory):** Recall relevant historical details or traditional knowledge.
3.  **Yukti (Logical Integration):** Integrate Drishti and Smriti to logically rule out inconsistent options.
4.  **Sthiti (Regional Contextualization):** Align the insights with specific regional and cultural contexts within India.

-----

## 🛠️ Dataset Creation Pipeline

<img width="1404" height="422" alt="image" src="https://github.com/user-attachments/assets/a13d6107-804c-4736-8fe7-a0f025ed89e1" />

The dataset was created using a four-stage process, culminating in 64,290 instances.

1.  **Knowledge Curation:** Created 2,126 original English MCQs with images and smart distractors from diverse cultural sources.
2.  **Cultural Tagging:** Categorized all questions using 16 fine-grained attributes (e.g., festivals, cuisine).
3.  **Reasoning Augmentation:** Generated over 2,160 new, challenging questions testing common sense, multi-hop, and analogy-based cultural reasoning.
4.  **Multilingual Scale-up:** Translated and human-verified the data into 14 Indic languages, resulting in the final benchmark of 64,288 instances.


### Example Question Snapshot

<img width="1085" height="489" alt="image" src="https://github.com/user-attachments/assets/7940f49b-f327-41b6-bd84-ccf5373c3300" />


-----

## 📈 Evaluation and Results

We benchmarked a wide range of state-of-the-art VLMs, including proprietary systems (e.g., GPT-4o-mini), open-source large/small models (e.g. InternVL3 , LLAVA), reasoning-specialized models (e.g., Kimi-VL), and Indic-aligned models (e.g., Chitrarth, Maya). The primary metric used was **Accuracy** in both Zero-shot and Chain-of-Thought (CoT) evaluation setups.

<img width="1336" height="628" alt="image" src="https://github.com/user-attachments/assets/6a02ad7c-1d44-48ca-8d23-49e52d19a99c" />


### Key Findings & Challenges

  * VLMs demonstrate **critical limitations** in reasoning over culturally grounded, multimodal inputs.
  * Significant **performance gaps** persist, particularly for low-resource languages (e.g., Konkani, Sindhi) and less-documented regional traditions, highlighting digital inequities.
  * **Chain-of-Thought (CoT) prompting** generally enhanced culturally grounded reasoning but showed inconsistent benefits across question types and languages.
  * **Error Analysis** revealed models struggled with:
      * **Fine-grained semantic confusion** when distractor options were semantically close to the correct answer.
      * **Over-reliance on lexical cues** rather than a comprehensive understanding of the context, especially in culturally nuanced questions.
      * **Gaps in visual grounding** where accurate interpretation required deeper regional or cultural knowledge.

-----

## 🔗 Citation

If you use DRISHTIKON in your research, please cite the corresponding paper.

```bibtex
@inproceedings{Maji2025Drishtikon,
  title={DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models' Understanding on Indian Culture},
  author={Arijit Maji, Raghvendra Kumar, Akash Ghosh, Anushka, Nemil Shah, Abhilekh Borah, Vanshika Shah, Nishant Mishra, Sriparna Saha},
  booktitle={EMNLP 2025},
  year={2025}
}
```
