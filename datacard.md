# Pidgin-to-English Translation Dataset (Sample)
## Data Card v1.0

**Dataset Name:** Pidgin-to-English Translation Dataset (Sample)  
**Dataset Type:** Sample Dataset  
**Version:** 1.0  
**Release Date:** 2026  
**Organization:** Bytte AI  
**License:** CC-BY-4.0  
**Contact:** contact@bytteai.xyz  
**Website:** https://www.bytte.xyz/

> **Note:** This is a **sample dataset** containing a representative subset of translation pairs. It is designed for initial exploration, prototyping, and demonstrating translation quality for Nigerian Pidgin-English pairs.

---

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Dataset Composition](#dataset-composition)
- [Data Collection and Creation](#data-collection-and-creation)
- [Data Format](#data-format)
- [Quality Metrics](#quality-metrics)
- [Intended Use](#intended-use)
- [Limitations and Risks](#limitations-and-risks)
- [Access and Distribution](#access-and-distribution)
- [Citation](#citation)

---

## Dataset Overview

The Pidgin-to-English Translation Dataset is a conversational-style translation corpus containing 122 Pidgin-English to Standard English translation pairs. This **sample dataset** is designed to support machine translation research, low-resource language processing, and cross-lingual understanding between Nigerian Pidgin and English.

**Sample Dataset Characteristics:** This dataset represents a curated sample of translation pairs demonstrating typical Pidgin-English conversational patterns and translation challenges. It showcases authentic Pidgin grammatical features and provides a foundation for prototyping translation systems.

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Translation Pairs** | 122 |
| **Source Language** | Nigerian Pidgin English |
| **Target Language** | Standard English |
| **Format** | Conversational JSON (user/assistant pairs) |
| **Average Source Length** | 10.75 words (53 characters) |
| **Average Target Length** | 16.16 words (87 characters) |
| **Length Expansion Ratio** | 1.56x (English is ~56% longer) |
| **Domain** | Conversational/everyday language |

---

## Dataset Composition

### Translation Pairs

- **Format:** Conversational structure with user (Pidgin) and assistant (English) roles
- **Sentence Complexity:** Simple to moderate (5-25 words in Pidgin source)
- **Domain Coverage:** Everyday conversations, common phrases, social interactions
- **Style:** Mix of direct translation and conversational responses

### Sample Distribution

| Characteristic | Count | Percentage |
|----------------|-------|------------|
| Total pairs | 122 | 100% |
| Source with Pidgin markers (dey, don, etc.) | 117 | 95.9% |
| Pure direct translations | ~103 | 84.4% |
| Conversational expansions | ~10 | 8.2% |
| Targets with Pidgin code-switching | 9 | 7.4% |

### Linguistic Characteristics

**Common Pidgin Features in Source Texts:**

| Feature | Occurrences | % of Samples | Meaning/Usage |
|---------|-------------|--------------|---------------|
| **dey** | 56 | 45.9% | Continuous aspect marker ("is/are -ing") |
| **no** | 34 | 27.9% | Negation marker ("not", "don't") |
| **go** | 33 | 27.0% | Future marker or movement verb |
| **make** | 15 | 12.3% | Causative/let, subjunctive marker |
| **don** | 13 | 10.7% | Completive aspect marker ("have/has") |
| **wey** | 11 | 9.0% | Relative pronoun ("that", "which") |
| **am** | 8 | 6.6% | Object pronoun ("him/her/it") |
| **wetin** | 8 | 6.6% | Question word ("what") |
| **abeg** | 6 | 4.9% | Politeness marker ("please") |

These markers demonstrate authentic Nigerian Pidgin grammatical features including aspect marking, serial verb constructions, and unique pronouns.

---

## Data Collection and Creation

### Source

The dataset consists of translation pairs created through conversational interactions with AI chatbots. These translations represent common Nigerian Pidgin expressions and their English equivalents generated through:

- AI-assisted conversation and translation
- Conversational exchanges modeling typical Pidgin usage
- Human review and validation of AI-generated pairs

### Creation Methodology

- **Approach:** Conversational translation format generated through AI chatbot interactions
- **Structure:** JSON objects with user/assistant conversation pairs
- **Quality Control:** Human review and validation of AI-generated translations
- **Annotation:** Single category assignment per sample

### Data Characteristics

**Source Text (Pidgin):**
- Average length: 10.75 words
- Range: 5-25 words per sentence
- Character count: 20-148 characters
- Authentic Pidgin grammatical structures

**Target Text (English):**
- Average length: 16.16 words
- Range: 4-35 words per sentence
- Character count: 24-191 characters
- Mix of direct translations and conversational responses

---

## Data Format

### File Structure

**Filename:** `pidgin_to_english_translation.json`  
**Size:** 40 KB  
**Format:** JSON array of conversation objects

### Schema

```json
[
  {
    "conversations": [
      {
        "role": "user",
        "content": "The politician dey promise change, but the people no believe am."
      },
      {
        "role": "assistant",
        "content": "The politician is promising change, but the people do not believe him."
      }
    ],
    "category": "pidgin_to_english_translation",
    "category_description": "Simple Pidgin → English translation"
  }
]
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `conversations` | Array | List of conversation turns (always 2 items) |
| `conversations[0].role` | String | Always "user" (Pidgin source) |
| `conversations[0].content` | String | Pidgin input sentence |
| `conversations[1].role` | String | Always "assistant" (English target) |
| `conversations[1].content` | String | English translation or response |
| `category` | String | Always "pidgin_to_english_translation" |
| `category_description` | String | Task description |

---

## Quality Metrics

### 1. Translation Consistency

**Metric:** Percentage of samples with direct, accurate translations  
**Score:** 84.43%

**Breakdown:**
- Direct translations (high quality): ~103 samples (84.4%)
- Conversational expansions: ~10 samples (8.2%)
- Code-switched targets (Pidgin in English): 9 samples (7.4%)

**Interpretation:** The majority of translations are direct and accurate. A minority contain conversational expansions or retain Pidgin elements in the target.

### 2. Length Ratio Variance

**Metric:** Variance in translation length ratios (English words / Pidgin words)  
**Score:** 0.8353

**Statistics:**
- Average ratio: 1.563
- Median ratio: 1.167
- Range: 0.400 – 5.200

**Interpretation:** Most translations expand by 50-60%, but some show extreme variation due to conversational responses. Lower variance would indicate more consistent direct translation.

### 3. Pidgin Authenticity

**Metric:** Percentage of source texts containing authentic Pidgin grammatical markers  
**Score:** 95.9%

**Key Markers Detected:**
- Aspect markers: "dey" (continuous), "don" (completive)
- Negation: "no" 
- Pronouns: "am", "wetin"
- Discourse markers: "abeg", "make"

**Interpretation:** Source texts demonstrate high linguistic authenticity with characteristic Pidgin features.

### 4. Average Labels Per Item

**Metric:** Number of translations per source sentence  
**Score:** 1.0

All source sentences have exactly one target translation. No alternative translations or multiple references are provided.

### 5. Domain Coverage

**Primary Domain:** Conversational/everyday language (100%)

**Topic Distribution (estimated):**
- Social interactions: ~35%
- Daily activities: ~25%
- Questions and requests: ~20%
- Statements and observations: ~20%

**Style Variation:**
- Direct translation: ~84%
- Conversational response: ~16%

---

## Intended Use

### Primary Use Cases

1. **Machine Translation Research**
   - Train Pidgin-to-English translation models
   - Evaluate neural machine translation (NMT) systems
   - Benchmark translation quality for low-resource languages
   - Develop conversational AI for Nigerian contexts

2. **Low-Resource NLP Development**
   - Bootstrap translation systems with limited parallel data
   - Fine-tune multilingual models on Pidgin-English pairs
   - Study transfer learning from English to Pidgin
   - Investigate code-switching phenomena

3. **Cross-Lingual Understanding**
   - Align Pidgin and English representations
   - Develop cross-lingual embeddings
   - Support information retrieval across languages
   - Enable bilingual applications

4. **Educational Applications**
   - Language learning tools for Pidgin speakers
   - Translation assistance for non-Pidgin speakers
   - Cultural and linguistic documentation
   - Pidgin language standardization efforts

### Recommended Applications

✅ **Suitable for:**
- Fine-tuning pre-trained translation models
- Evaluation benchmarks for Pidgin NLP
- Augmentation with back-translation
- Conversational AI training data
- Research on code-switching and language contact
- Educational tools and language resources

### Out-of-Scope Uses

❌ **Not recommended for:**
- Training translation models from scratch (insufficient scale)
- Production translation systems without validation (quality issues present)
- Applications requiring perfectly parallel translations (some conversational expansions)
- Formal or technical translation (conversational domain only)
- As sole training data (combine with other resources)

---

## Limitations and Risks

### Dataset Limitations

#### 1. **Code-Switching in Target Translations**

**Issue:** 9 samples (7.4%) contain Pidgin phrases or markers in the English target text.

**Examples:**
```
Pidgin:  I go dey your house by 5pm, no wahala.
English: Sure, no wahala. I'll be expecting you around 5pm. Safe journey!

Pidgin:  Wetin dey for supper tonight? Abeg talk am for better Naija.
English: Tonight for supper, we go chop Jollof rice with fried plantain...
```

**Impact:** Models trained on this data may:
- Learn to produce code-switched outputs
- Fail to fully translate Pidgin elements
- Reproduce informal expressions inappropriately

**Mitigation:**
- Flag or filter these 9 samples for critical applications
- Use as positive examples for code-switching research
- Apply post-processing to detect and handle Pidgin retention
- Combine with cleaner parallel corpora

#### 2. **Conversational Expansion vs. Direct Translation**

**Issue:** ~10 samples (8.2%) contain conversational responses rather than direct translations, with significant length expansion (>80% longer than source).

**Examples:**
```
Pidgin:  Can you make a joke about Jollof rice that won't offend anyone?
English: Sure! Here's a light-hearted Jollof rice joke:
         Why did the Jollof rice break up with fried rice?
         Because it just couldn't handle the heat of the competition!
```

**Impact:**
- Inconsistent translation style across dataset
- May confuse sequence-to-sequence models expecting 1:1 correspondence
- Inflates length ratios and complicates alignment

**Mitigation:**
- Document samples with expansion ratios >2.0x
- Use style tags or metadata to distinguish translation types
- Train separate models for direct translation vs. conversational response
- Apply length constraints during inference

#### 3. **Limited Scale (Sample Dataset)**

**Size:** 122 translation pairs  
**Status:** This is a sample dataset

**Impact:** 
- Insufficient for training translation models from scratch
- Represents a curated sample, not the full corpus
- Best suited for prototyping and fine-tuning

**Comparison to Standard MT Datasets:**
- WMT datasets: Millions of parallel sentences
- Low-resource MT: Often 10,000+ pairs minimum
- This sample dataset: 122 pairs (0.01% of typical scale)

**Mitigation:**
- Use only for fine-tuning pre-trained models
- Combine with synthetic data (back-translation, paraphrasing)
- Leverage multilingual models with transfer learning
- Augment with monolingual Pidgin/English data

#### 4. **Domain Specificity**

**Coverage:** Exclusively conversational/everyday language  
**Missing Domains:** News, technical, medical, legal, literary

**Impact:**
- Models may underperform on formal or specialized text
- Limited vocabulary coverage beyond conversational contexts
- May not capture domain-specific Pidgin terminology

**Mitigation:**
- Clearly document domain limitations
- Combine with domain-specific corpora when available
- Apply domain adaptation techniques for new contexts

#### 5. **Single Reference Translations**

**Issue:** Each Pidgin sentence has only one English translation  
**Impact:** Cannot measure translation diversity or provide alternative renderings

**Limitations:**
- No inter-annotator agreement metrics
- Cannot evaluate synonymous translations
- May miss valid alternative phrasings

**Mitigation:**
- Use BLEU, chrF, or other reference-based metrics cautiously
- Consider human evaluation for quality assessment
- Create multiple references for critical evaluation sets

#### 6. **Length Ratio Variability**

**Variance:** 0.8353 (high variability in translation length)  
**Range:** 0.4x to 5.2x expansion

**Impact:**
- Inconsistent alignment between source and target
- Challenges for attention mechanisms
- Difficult to predict output length

**Mitigation:**
- Apply length penalties during decoding
- Filter extreme ratios for training (e.g., exclude >3x)
- Use copy mechanisms for very short translations

#### 7. **Lack of Annotation Metadata**

**Missing Information:**
- No annotator IDs or timestamps
- No inter-annotator agreement scores
- No translation quality ratings
- No source provenance (original vs. constructed)

**Impact:**
- Cannot assess annotation reliability
- Cannot stratify by quality or difficulty
- Limited error analysis capability

**Mitigation:**
- Conduct post-hoc quality assessment
- Manual review of samples for critical applications
- Establish validation sets with gold-standard annotations

### Potential Risks

#### 1. **Propagation of Translation Errors**

**Risk:** Models trained on this data may learn incorrect translation patterns from the 15.6% of problematic samples.

**Examples:**
- Retaining Pidgin in English outputs
- Over-expanding simple translations
- Producing conversational flourishes instead of direct translations

**Mitigation:**
- Manually review and clean problematic samples
- Use data filtering based on quality metrics
- Implement confidence thresholds during inference

#### 2. **Reinforcement of Informal Language**

**Risk:** Conversational style may be inappropriate for formal contexts.

**Impact:**
- Translations may lack formality for professional settings
- Models may produce overly casual outputs
- Code-switching patterns may be reinforced

**Mitigation:**
- Document intended use cases (conversational only)
- Fine-tune separate models for formal vs. informal contexts
- Apply style transfer techniques post-translation

#### 3. **Limited Linguistic Diversity**

**Risk:** Dataset may not represent all varieties of Nigerian Pidgin.

**Concerns:**
- Regional dialects not covered
- Age-specific slang missing
- Evolving Pidgin features absent

**Mitigation:**
- Combine with diverse Pidgin sources
- Acknowledge geographic and demographic limitations
- Update dataset periodically with new variations

#### 4. **Evaluation Challenges**

**Risk:** Quality metrics may be misleading due to single references and style inconsistency.

**Issues:**
- BLEU scores may underestimate quality (one reference)
- Automatic metrics can't detect code-switching errors
- Conversational expansions inflate scores artificially

**Mitigation:**
- Use multiple evaluation metrics (BLEU, METEOR, chrF, BERTScore)
- Conduct human evaluation for critical assessments
- Report metric limitations in publications

### Recommended Best Practices

✅ **Data Cleaning:**
1. Identify and flag 9 samples with Pidgin in English targets
2. Mark 10 samples with excessive conversational expansion
3. Consider filtering or reweighting these 19 samples (15.6%)

✅ **Training Strategies:**
1. Use as fine-tuning data only (not pre-training)
2. Combine with larger parallel corpora (e.g., JW300 if available)
3. Apply data augmentation (back-translation, paraphrasing)
4. Use stratified splits to maintain quality distribution

✅ **Evaluation Protocols:**
1. Report multiple metrics (BLEU, chrF, BERTScore)
2. Include human evaluation for publication-quality work
3. Test on held-out data from different sources
4. Analyze errors by translation type (direct vs. conversational)

✅ **Documentation:**
1. Cite known limitations when publishing results
2. Report which samples were filtered (if any)
3. Acknowledge dataset size constraints
4. Provide context on conversational vs. direct translation style

---

## Access and Distribution

### Download Locations

- **Hugging Face:** https://huggingface.co/datasets/Bytte-AI/Pidgin-to-English-conversational-translations
- **Figshare:** https://figshare.com/articles/dataset/_b_Pidgin-to-English-translations_b_/31259068

### File Information

| File | Format | Size | Description |
|------|--------|------|-------------|
| `pidgin_to_english_translation.json` | JSON | 40 KB | 122 Pidgin-English translation pairs |

### License

**CC-BY-4.0 (Creative Commons Attribution 4.0 International)**

You are free to:
- ✅ **Share** — copy and redistribute the material
- ✅ **Adapt** — remix, transform, and build upon the material
- ✅ **Commercial use** — use for commercial purposes

Under the following terms:
- 📌 **Attribution** — You must give appropriate credit to Bytte AI, provide a link to the license, and indicate if changes were made

### Terms of Use

1. **Attribution Required:** Cite this dataset using the provided citation format
2. **Acknowledge Limitations:** Clearly document known issues (code-switching, conversational expansions) in publications
3. **Quality Filtering Recommended:** Consider filtering the 19 flagged samples for critical applications
4. **No Warranty:** Provided "as-is" without guarantees of translation accuracy

---

## Citation

If you use this dataset in your research or applications, please cite:

```bibtex
@dataset{bytte_ai_pidgin_english_translation_2026,
  author    = {Bytte AI},
  title     = {Pidgin-to-English Translation Dataset (Sample)},
  year      = {2026},
  version   = {1.0},
  note      = {Sample dataset - AI chatbot-generated translations},
  publisher = {Hugging Face and Figshare},
  url       = {https://huggingface.co/datasets/Bytte-AI/Pidgin-to-English-conversational-translations},
  license   = {CC-BY-4.0}
}
```

**APA Format:**
```
Bytte AI. (2026). Pidgin-to-English Translation Dataset (Sample) (Version 1.0) [Data set]. 
Hugging Face. https://huggingface.co/datasets/Bytte-AI/Pidgin-to-English-conversational-translations
``` 
Part of BBC Igbo–Pidgin Gold-Standard NLP Corpus. Hugging Face. 
https://huggingface.co/datasets/Bytte-AI/BBC_Igbo-Pidgin_Gold-Standard_NLP_Corpus
```

---

## Contact and Support

**Organization:** Bytte AI  
**Email:** contact@bytteai.xyz  
**Website:** https://www.bytte.xyz/

For questions, feedback, or to report data quality issues, please contact us via email.

---

## Acknowledgments

This translation dataset was created by Bytte AI through AI chatbot interactions with human validation. We acknowledge the importance of Nigerian Pidgin as a vital language for millions of West Africans and the need for high-quality linguistic resources to support digital inclusion and NLP research.

---

## Version History

**v1.0 (2026)**
- Initial release
- 122 Pidgin-to-English translation pairs
- Conversational format (user/assistant)
- Known limitations documented (code-switching, conversational expansions)

---

## Appendix: Sample Quality Issues

### Samples with Pidgin in English Target (9 total)

| Index | Pidgin Source | English Target (Issue Highlighted) |
|-------|---------------|-----------------------------------|
| 4 | Wetin dey for supper tonight? | ...we **go chop** Jollof rice...I sure say this...go sweet **well well, no wahala!** |
| 9 | You fit help me with this wahala? | ...I'll do my best to assist you. **No wahala** at all! |
| 11 | I go dey your house by 5pm, no wahala. | Sure, **no wahala**. I'll be expecting you... |

### Samples with Excessive Conversational Expansion (10 total)

| Index | Pidgin Source (10 words) | English Target (30+ words) |
|-------|-------------------------|---------------------------|
| 5 | Can you make a joke about Jollof rice... | Sure! Here's a light-hearted Jollof rice joke: [full joke structure] |
| 26 | I no fit come today, abeg. | I understand. If you can reschedule for another time that is convenient... |

*These samples may require filtering or special handling depending on use case.*

---

**Last Updated:** February 2026  
**Maintained by:** Bytte AI Research Team
