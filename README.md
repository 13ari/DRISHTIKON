## DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models' Understanding on Indian Culture

### Overview

DRISHTIKON is a **first-of-its-kind multimodal, multilingual benchmark** dedicated to evaluating Vision-Language Models’ (VLMs) ability to understand Indian culture. Unlike existing global benchmarks, DRISHTIKON focuses exclusively on India’s cultural richness, spanning **15 languages, all 28 states, and 8 union territories**, with **64,288 carefully curated text-image pairs**.

The dataset captures **festivals, attire, cuisines, rituals, art forms, architecture, personalities, and heritage**, enabling **fine-grained evaluation of cultural reasoning** in multimodal models.

* **Paper:** [DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models' Understanding on Indian Culture](arxiv.org/abs/2509.19274)
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

```@inproceedings{Maji2025Drishtikon,
  title={DRISHTIKON: A Multimodal Multilingual Benchmark for Testing Language Models' Understanding on Indian Culture},
  author={Arijit Maji and Raghvendra Kumar and Akash Ghosh and Anushka and Nemil Shah and Abhilekh Borah and Vanshika Shah and Nishant Mishra and Sriparna Saha},
  booktitle={EMNLP 2025},
  year={2025}
}```
