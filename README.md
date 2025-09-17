## DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models' Understanding on Indian Culture

### Overview

DRISHTIKON is a **first-of-its-kind multimodal, multilingual benchmark** dedicated to evaluating Vision-Language Models’ (VLMs) ability to understand Indian culture. Unlike existing global benchmarks, DRISHTIKON focuses exclusively on India’s cultural richness, spanning **15 languages, all 28 states, and 8 union territories**, with **64,288 carefully curated text-image pairs**.

The dataset captures **festivals, attire, cuisines, rituals, art forms, architecture, personalities, and heritage**, enabling **fine-grained evaluation of cultural reasoning** in multimodal models.

* **Paper:** [EMNLP 2024](https://doi.org/10.18653/v1/2024.emnlp-main.882)
* **Dataset:** [Hugging Face](https://huggingface.co/datasets/13ari/DRISHTIKON)

---

### ✨ Key Features

* **Scale:** 64K+ text-image pairs.
* **Languages:** 15 (including English + 14 major Indian languages).
* **Coverage:** All 28 states and 8 union territories.
* **Images:** Collected from **public websites** and provided as **zipped archives** on Hugging Face.
* **Question Types:**

  * General questions
  * Cultural commonsense questions
  * Multi-hop reasoning questions
  * Analogy-based questions

---

### Dataset Structure

Each entry contains:

* Culturally grounded **question** (MCQ format with 4 options).
* **Correct answer label**.
* **Associated image (stored in zip, path provided)**.
* Metadata: language, state/UT, cultural category (attire, cuisine, rituals, etc.), question type.

---

### 📊 Uses

**Direct Use:**

* Benchmarking VLMs for cultural understanding.
* Evaluating multilingual multimodal reasoning.
* Comparative analysis of open-source vs proprietary systems.

**Out-of-Scope Use:**

* Commercial applications.
* Misuse to reinforce stereotypes or generate culturally insensitive content.

---

### ⚙️ Dataset Creation

* **Data Sources:** Wikipedia, Holidify, Ritiriwaz, Google Arts & Culture, Times of India, and others.
* **MCQs:** 2,126 base questions, extended with 2,160 reasoning-augmented MCQs.
* **Multilingual Expansion:** Translated into 14 Indian languages with Gemini Pro, human-verified.
* **Annotations:** Manual tagging of cultural categories, peer-reviewed & expert-adjudicated.
* **Images:** Acquired from **public websites**; distributed as zipped archives for reproducibility.

---

### ⚠️ Bias, Risks, and Limitations

* Not exhaustive of all cultural nuances.
* Limited coverage for dialects and micro-traditions.
* May reflect annotator bias despite careful validation.
* Low-resource languages remain challenging for models.

---

### 📜 License

* Released for **research and non-commercial use only**.
* Images are included in **zipped format** (collected from public websites).
* Users must comply with original source licenses.

---

### ✍️ Citation

```bibtex
@inproceedings{maji2024drishtikon,
  title={DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models’ Understanding on Indian Culture},
  author={Maji, Arijit and Kumar, Raghvendra and Ghosh, Akash and Anushka, and Shah, Nemil and Borah, Abhilekh and Shah, Vanshika and Mishra, Nishant and Saha, Sriparna},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024}
}
```
