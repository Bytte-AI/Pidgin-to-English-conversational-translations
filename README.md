# Pidgin-to-English Translation Dataset (Sample)

<div align="center">

![Version](https://img.shields.io/badge/version-1.0-blue.svg)
![Type](https://img.shields.io/badge/type-Sample%20Dataset-purple.svg)
![License](https://img.shields.io/badge/license-CC--BY--4.0-green.svg)
![Language](https://img.shields.io/badge/language-Pidgin%20→%20English-orange.svg)
![Pairs](https://img.shields.io/badge/pairs-122-brightgreen.svg)
![Quality](https://img.shields.io/badge/consistency-84.43%25-yellow.svg)

**Sample dataset: Nigerian Pidgin to English translation pairs for machine translation research**

[🤗 Hugging Face](https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus) • [📊 Figshare](https://doi.org/10.6084/m9.figshare.31259068) • [🌐 Website](https://www.bytte.xyz/) • [📧 Contact](mailto:contact@bytteai.xyz)

</div>

---

## 📋 Overview

The **BBC Pidgin-to-English Translation Dataset (Sample)** is a conversational-style parallel corpus containing 122 translation pairs from Nigerian Pidgin English to Standard English. Created by **Bytte AI**, this sample dataset supports machine translation research, low-resource NLP, and cross-lingual understanding for African languages.

> **📌 Sample Dataset Notice:** This is a **sample dataset** with 122 curated translation pairs. It demonstrates authentic Pidgin features and translation challenges, ideal for prototyping, fine-tuning, and initial MT research.

### 🎯 Key Features

- **122 translation pairs** in conversational format
- **Authentic Nigerian Pidgin** with characteristic grammatical markers
- **Conversational style** suitable for dialogue systems
- **Everyday language domain** covering social interactions
- **Documented quality issues** for transparent use
- **Part of larger corpus** with complementary NLP datasets

### 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Translation Pairs** | 122 |
| **Avg Source Length** | 10.75 words (Pidgin) |
| **Avg Target Length** | 16.16 words (English) |
| **Length Ratio** | 1.56x expansion |
| **Translation Consistency** | 84.43% |
| **Pidgin Authenticity** | 95.9% (with markers) |
| **Domain** | Conversational/everyday |

---

## 🗂️ Dataset Composition

### Translation Characteristics

```
Source (Pidgin):  The politician dey promise change, but the people no believe am.
Target (English): The politician is promising change, but the people do not believe him.

Source (Pidgin):  He don chop finish before you call am.
Target (English): He had already finished eating before you called him.
```

### Distribution Breakdown

| Category | Count | % | Description |
|----------|-------|---|-------------|
| **Direct translations** | ~103 | 84.4% | Clean Pidgin → English translations |
| **Conversational expansions** | ~10 | 8.2% | Responses longer than simple translation |
| **Code-switched targets** | 9 | 7.4% | English output contains Pidgin phrases |
| **Total pairs** | 122 | 100% | All translation samples |

### Linguistic Features

**Most Common Pidgin Markers:**

| Marker | Count | % | Function |
|--------|-------|---|----------|
| **dey** | 56 | 45.9% | Continuous aspect ("is/are -ing") |
| **no** | 34 | 27.9% | Negation ("not", "don't") |
| **go** | 33 | 27.0% | Future tense or movement |
| **make** | 15 | 12.3% | Subjunctive ("let", "should") |
| **don** | 13 | 10.7% | Perfect aspect ("have/has") |
| **wey** | 11 | 9.0% | Relative pronoun ("which") |

---

## 🚀 Getting Started

### Installation

```bash
# Clone the corpus repository
git clone https://github.com/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus.git
cd BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus

# Install dependencies
pip install datasets transformers
```

### Quick Load

```python
import json

# Load translation data
with open('pidgin_to_english_translation.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total translation pairs: {len(data)}")

# Extract first example
example = data[0]
pidgin = example['conversations'][0]['content']
english = example['conversations'][1]['content']

print(f"Pidgin:  {pidgin}")
print(f"English: {english}")
```

**Output:**
```
Total translation pairs: 122
Pidgin:  The politician dey promise change, but the people no believe am.
English: The politician is promising change, but the people do not believe him.
```

### Load with Hugging Face Datasets

```python
from datasets import load_dataset

# Load full corpus (includes translation dataset)
corpus = load_dataset("Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus")

# Access translation subset
translation_data = corpus['pidgin_english_translation']
```

### Prepare for Training

```python
import json
from sklearn.model_selection import train_test_split

# Load data
with open('pidgin_to_english_translation.json', 'r') as f:
    data = json.load(f)

# Extract parallel sentences
source_sentences = [item['conversations'][0]['content'] for item in data]
target_sentences = [item['conversations'][1]['content'] for item in data]

# Split data (80/20 train/test)
src_train, src_test, tgt_train, tgt_test = train_test_split(
    source_sentences, target_sentences, 
    test_size=0.2, 
    random_state=42
)

print(f"Training pairs: {len(src_train)}")
print(f"Test pairs: {len(src_test)}")
```

### Fine-tune mBART for Translation

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load pre-trained model
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")

# Set source and target languages
tokenizer.src_lang = "en_XX"  # Use English as closest to Pidgin
tokenizer.tgt_lang = "en_XX"

# Prepare data
def prepare_data(src, tgt):
    inputs = tokenizer(src, padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(tgt, padding=True, truncation=True, return_tensors="pt").input_ids
    return {"input_ids": inputs.input_ids, "labels": labels}

# Fine-tune on your Pidgin-English pairs
# ... (standard training loop)
```

### Example: Translation Inference

```python
# After training your model
def translate_pidgin_to_english(pidgin_text, model, tokenizer):
    inputs = tokenizer(pidgin_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Test translation
pidgin = "I dey come your house tomorrow."
english = translate_pidgin_to_english(pidgin, model, tokenizer)
print(f"Pidgin:  {pidgin}")
print(f"English: {english}")
```

---

## 📈 Quality Metrics

### Translation Consistency: 84.43%

```
✅ Clean translations:     103 samples (84.4%)
⚠️  Conversational style:  10 samples (8.2%)
⚠️  Code-switched targets: 9 samples (7.4%)
```

**Interpretation:** Most translations are direct and accurate, but ~16% have quality issues.

### Length Statistics

| Metric | Pidgin (Source) | English (Target) | Ratio |
|--------|-----------------|------------------|-------|
| **Average** | 10.75 words | 16.16 words | 1.56x |
| **Median** | — | — | 1.17x |
| **Range** | 5-25 words | 4-35 words | 0.4-5.2x |
| **Variance** | — | — | 0.8353 |

**Note:** High variance indicates inconsistent translation style (direct vs. conversational).

### Pidgin Authenticity: 95.9%

117 of 122 samples contain authentic Pidgin grammatical markers, demonstrating linguistic validity.

---

## ⚠️ Known Limitations

### 🔴 Critical: Code-Switching in Targets (7.4%)

**Issue:** 9 samples contain Pidgin phrases in English translations.

**Example:**
```
Pidgin:  I go dey your house by 5pm, no wahala.
English: Sure, no wahala. I'll be expecting you around 5pm. ← Contains "no wahala"
```

**Impact:** Models may learn to produce partially-translated outputs.

**Mitigation:**
- Filter these 9 samples for critical applications
- Use for code-switching research instead
- List of affected indices: `[4, 9, 11, 12, 19, 34, 36, 95, ...]`

### 🟡 Moderate: Conversational Expansions (8.2%)

**Issue:** ~10 samples are conversational responses, not direct translations.

**Example:**
```
Pidgin:  Can you make a joke about Jollof rice...?
English: Sure! Here's a light-hearted Jollof rice joke:
         Why did the Jollof rice break up with fried rice?
         Because it just couldn't handle the heat...
```

**Impact:** Inflates length ratios; may confuse seq2seq models.

**Mitigation:**
- Mark samples with length ratio >2.0x
- Train separate models for direct translation vs. dialogue
- Apply length constraints during inference

### 🟡 Moderate: Limited Scale (Sample Dataset - 122 pairs)

**Status:** This is a sample dataset

**Comparison:**
- This sample: 122 pairs
- Typical low-resource MT: 10,000+ pairs
- WMT datasets: Millions of pairs

**Impact:** Insufficient for training from scratch; best for fine-tuning.

**Mitigation:**
- ✅ Use for fine-tuning only
- ✅ Combine with data augmentation (back-translation)
- ✅ Leverage multilingual pre-trained models
- ❌ Don't train translation models from scratch

### 🟢 Minor: Domain Specificity

**Coverage:** Only conversational/everyday language  
**Missing:** News, technical, medical, legal domains

**Mitigation:** Combine with domain-specific data when available.

---

## 💡 Use Cases

### ✅ Recommended Uses

1. **Machine Translation Research**
   - Fine-tune mBART, mT5, or NLLB models
   - Benchmark translation quality for Pidgin
   - Study low-resource translation techniques

2. **Conversational AI**
   - Train chatbots for Nigerian audiences
   - Develop Pidgin-English assistants
   - Support bilingual dialogue systems

3. **Code-Switching Research**
   - Analyze Pidgin-English mixing patterns
   - Study language contact phenomena
   - Investigate bilingual processing

4. **Educational Tools**
   - Language learning applications
   - Translation assistance for learners
   - Pidgin standardization resources

### ❌ Not Recommended

- Training translation models from scratch (too small)
- Production systems without validation (quality issues present)
- Formal/technical translation (conversational domain only)
- As sole evaluation benchmark (single references, style variation)

---

## 🛠️ Data Cleaning Recommendations

### Option 1: Use All Data (Default)

```python
# Load all 122 samples
with open('pidgin_to_english_translation.json', 'r') as f:
    data = json.load(f)
```

**Pros:** Maximum data for training  
**Cons:** Includes 19 problematic samples  
**Best for:** Data augmentation, code-switching research

### Option 2: Filter Problematic Samples

```python
# Manually exclude indices with known issues
EXCLUDE_INDICES = [4, 5, 9, 11, 12, 19, 26, 34, 36, 95, ...]  # 19 total

clean_data = [item for i, item in enumerate(data) if i not in EXCLUDE_INDICES]
print(f"Clean samples: {len(clean_data)}")  # ~103 samples
```

**Pros:** Higher quality translations  
**Cons:** Reduces dataset size by 15.6%  
**Best for:** Critical applications, benchmarking

### Option 3: Stratified Splits

```python
# Separate by quality tier
direct_translations = []      # High quality
conversational_expansions = [] # Medium quality  
code_switched = []             # Low quality (for research)

for i, item in enumerate(data):
    if i in CODE_SWITCHED_INDICES:
        code_switched.append(item)
    elif i in CONVERSATIONAL_INDICES:
        conversational_expansions.append(item)
    else:
        direct_translations.append(item)
```

**Best for:** Ablation studies, quality analysis

---

## 📊 Evaluation Guidelines

### Recommended Metrics

```python
from sacrebleu import corpus_bleu
from bert_score import score as bert_score

# BLEU (with caution - single reference)
bleu = corpus_bleu(predictions, [references])

# BERTScore (semantic similarity)
P, R, F1 = bert_score(predictions, references, lang='en')

# chrF (character-level)
chrf = corpus_chrf(predictions, [references])
```

⚠️ **Important:** Single-reference BLEU may underestimate quality. Use multiple metrics.

### Human Evaluation

For publication-quality work, include human assessment:

- **Adequacy:** Does translation preserve meaning?
- **Fluency:** Is English output natural?
- **Code-switching:** Any Pidgin retained inappropriately?

### Error Analysis

```python
# Analyze by length ratio
for pred, ref in zip(predictions, references):
    ratio = len(pred.split()) / len(ref.split())
    if ratio > 2.0:
        print(f"Expansion detected: {ratio:.2f}x")
        print(f"Ref: {ref}")
        print(f"Pred: {pred}\n")
```

---

## 📖 Data Format

### File Structure

```json
[
  {
    "conversations": [
      {
        "role": "user",
        "content": "Pidgin sentence here"
      },
      {
        "role": "assistant", 
        "content": "English translation here"
      }
    ],
    "category": "pidgin_to_english_translation",
    "category_description": "Simple Pidgin → English translation"
  }
]
```

### Accessing Fields

```python
for item in data:
    pidgin = item['conversations'][0]['content']   # Source
    english = item['conversations'][1]['content']  # Target
    category = item['category']                    # Task type
    
    print(f"{pidgin} → {english}")
```

---

## 📚 Citation

```bibtex
@dataset{bytte_ai_pidgin_english_translation_2026,
  author    = {Bytte AI},
  title     = {BBC Pidgin-to-English Translation Dataset (Sample)},
  year      = {2026},
  version   = {1.0},
  note      = {Sample dataset - Part of BBC Igbo–Pidgin Gold-Standard NLP Corpus},
  publisher = {Hugging Face and Figshare},
  url       = {https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus},
  license   = {CC-BY-4.0}
}
```

---

## 📜 License

**CC-BY-4.0** - Free to use with attribution to Bytte AI.

---

## 🤝 Contributing

### Report Issues

Found translation errors or quality problems? Help us improve:

1. **Open an issue** on GitHub
2. **Specify sample index** and describe the problem
3. **Suggest correction** if possible

### Quality Improvements

We welcome contributions:
- Additional validation of problematic samples
- Alternative translations for single-reference pairs
- Annotation of translation quality scores
- Expansion to new domains

---

## 🌍 Related Resources

### Other Pidgin/African Language Datasets

- **JW300** - Multilingual parallel corpus (may include Pidgin)
- **FLORES-200** - Evaluation benchmark (limited African languages)
- **MasakhaNER** - NER for African languages
- **AfriSenti** - Sentiment analysis datasets

### Translation Models

- **mBART** - Multilingual BART for translation
- **mT5** - Multilingual T5
- **NLLB-200** - No Language Left Behind (200 languages)
- **M2M-100** - Many-to-many translation

---

## 📞 Contact

**Organization:** Bytte AI  
**Website:** https://www.bytte.xyz/  
**Email:** contact@bytteai.xyz

**Download:**
- 🤗 Hugging Face: https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus
- 📊 Figshare: https://figshare.com/articles/dataset/BBC_Igbo_Pidgin_Gold-Standard_NLP_Corpus/31249567

---

## 🙏 Acknowledgments

This dataset is part of the **BBC Igbo–Pidgin Gold-Standard NLP Corpus** created by Bytte AI. We acknowledge the importance of Nigerian Pidgin as a vital language for communication across West Africa and the need for quality resources to support NLP research and digital inclusion.

---

## 📅 Version History

### v1.0 (February 2026)
- Initial release
- 122 Pidgin-to-English translation pairs
- Conversational format
- Documented quality issues (code-switching, expansions)

---

## 🔮 Future Work

Potential improvements for future versions:

- ✨ Additional direct translations (target: 500+ pairs)
- ✨ Multi-reference translations for evaluation
- ✨ Quality tier annotations
- ✨ Domain expansion (news, social media)
- ✨ Cleaning of code-switched samples
- ✨ Inter-annotator agreement scores

---

## 📋 Quick Reference

### By the Numbers

| Metric | Value |
|--------|-------|
| Total pairs | 122 |
| Clean translations | ~103 (84.4%) |
| Problematic samples | ~19 (15.6%) |
| Avg source words | 10.75 |
| Avg target words | 16.16 |
| File size | 40 KB |

### Quality Tiers

- 🟢 **High:** Direct translations (84.4%)
- 🟡 **Medium:** Conversational style (8.2%)
- 🔴 **Low:** Code-switched (7.4%)

### Recommended Pipeline

1. Load data → 2. Filter if needed → 3. Split train/test → 4. Fine-tune model → 5. Evaluate with multiple metrics → 6. Human validation

---

<div align="center">

**Part of the BBC Igbo–Pidgin Gold-Standard NLP Corpus**

By [Bytte AI](https://www.bytte.xyz/) for African language NLP

[![License](https://img.shields.io/badge/license-CC--BY--4.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Hugging Face](https://img.shields.io/badge/🤗-Datasets-yellow.svg)](https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus)

</div>
