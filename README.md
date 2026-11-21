# Spoken Language Identification with CNNs

> 🚧 **Work in progress.** Part 1 (data pipeline) is done. Model training is next.

A deep learning project that identifies **which language is being spoken** in a short audio clip. The audio is converted into **log-mel spectrograms** (image-like representations of sound), and a **Convolutional Neural Network (CNN)** classifies them, turning an audio problem into an image classification problem.

---

## 🌍 Languages

The model is trained to recognize 9 languages from different language families:

| Label | Code | Language | Family |
|:---:|---|---|---|
| 0 | `es_419` | Spanish (Latin America) | Romance |
| 1 | `pt_br` | Portuguese (Brazil) | Romance |
| 2 | `fr_fr` | French | Romance |
| 3 | `ru_ru` | Russian | Slavic |
| 4 | `ja_jp` | Japanese | Japonic |
| 5 | `cmn_hans_cn` | Mandarin Chinese | Sino-Tibetan (tonal) |
| 6 | `ko_kr` | Korean | Koreanic |
| 7 | `en_us` | English (US) | Germanic |
| 8 | `nb_no` | Norwegian (Bokmål) | Germanic |

The set includes some deliberately **hard pairs**: Spanish vs Portuguese (closely related) and Japanese vs Korean (often confused by models), plus Mandarin as the only **tonal** language.

---

## 📦 Dataset

**[Google FLEURS](https://huggingface.co/datasets/google/fleurs)**: read speech in 100+ languages, 16 kHz audio, licensed under **CC-BY-4.0**.

- FLEURS provides official **train / validation / test** splits for every language. These are used as-is (not re-shuffled) to avoid the same speaker appearing in both training and test data.
- The data is **streamed** from the Hugging Face Hub, so the full dataset (~878 GB for all languages) never needs to be downloaded.

Available clips per split (smallest values highlighted):

| Language | Train | Validation | Test |
|---|---:|---:|---:|
| es_419 | 2796 | 408 | 908 |
| pt_br | 2793 | 386 | 919 |
| fr_fr | 3193 | 289 | 676 |
| ru_ru | 2562 | 356 | 775 |
| ja_jp | **2292** | 266 | 650 |
| cmn_hans_cn | 3246 | 409 | 945 |
| ko_kr | 2307 | 226 | 382 |
| en_us | 2602 | 394 | 647 |
| nb_no | 3167 | **163** | **357** |

To keep classes **balanced**, the same number of clips is taken from every language:

- **Current (development run):** 300 train / 50 validation / 50 test per language → 2,700 / 450 / 450 total
- **Planned (full run):** 2,000 / 160 / 350 per language, limited by the smallest split

---

## ⚙️ Pipeline (Part 1: Data Preparation)

### Step 1: Stream audio from FLEURS
Each language and split is loaded with `datasets.load_dataset(..., streaming=True)`, and the first *N* clips are taken with `.take(N)`.

### Step 2: Fix every clip to exactly 3 seconds
`crop_or_pad(audio)`
- At 16 kHz, 3 seconds = **48,000 samples**.
- Longer clips are cropped from the **middle**, because recordings often begin and end with silence, so the middle almost always contains speech.
- Shorter clips are padded with zeros (silence) at the end.
- CNNs need fixed-size inputs, so every clip must be the same length.

### Step 3: Convert audio into a log-mel spectrogram
`make_logmel(audio)`
- **Spectrogram:** shows which frequencies are present over time.
- **Mel scale:** compresses the frequency axis to match human hearing (more detail at low frequencies, where speech information lives).
- **Log (dB) scale:** makes quiet sounds visible next to loud ones.
- Output: a **64 × 94** "image":
  - 64 = mel frequency bands (height)
  - 94 = time steps (width)

### Step 4: Label and store
Each spectrogram is stored with its numeric label (0–8) in the matching split.

### Step 5: Save to Google Drive
All splits are saved as NumPy arrays so preprocessing only has to run once:

```
AudioCNNProject/
├── X_train.npy        # (2700, 64, 94)  spectrograms
├── y_train.npy        # (2700,)         labels 0–8
├── X_validation.npy   # (450, 64, 94)
├── y_validation.npy   # (450,)
├── X_test.npy         # (450, 64, 94)
└── y_test.npy         # (450,)
```

### Step 6: Visual check
A 3×3 grid shows one example spectrogram per language:

![Example spectrograms](images/spectrograms.png)

*How to read it:* left → right = time (0–3 s), bottom → top = frequency (low → high pitch), color = loudness (yellow = loud, purple = quiet). The horizontal stripes at the bottom are the **harmonics** of the speaker's voice, and dark vertical gaps are pauses between words.

> Note: these are single clips, so differences between them are partly due to the speaker and microphone, not only the language. The model learns from hundreds of clips per language to capture patterns that hold across many voices.

---

## 🗂️ Notebook Structure

`AudioCNNProject.ipynb`

1. **Imports and language list**
2. **Helper functions:** `crop_or_pad()` and `make_logmel()`
3. **Processing loop:** for each language → for each split → for each clip: crop, convert, label, store
4. **Visualization:** a single labeled spectrogram plus a 3×3 grid of all languages
5. **Saving:** export arrays to Google Drive

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**.
2. (For training) set **Runtime → Change runtime type → T4 GPU**.
3. Run all cells from top to bottom.
4. Allow Google Drive access when prompted. Data is saved to `MyDrive/AudioCNNProject/`.

No Hugging Face token is required (FLEURS is public), although setting `HF_TOKEN` in Colab Secrets gives faster downloads.

---

## 🛣️ Roadmap

- [x] Stream FLEURS audio for 9 languages
- [x] Crop / pad clips to 3 seconds
- [x] Convert to log-mel spectrograms
- [x] Visualize spectrograms per language
- [x] Save processed data to Google Drive
- [ ] Normalize spectrograms (mean/std from training set only)
- [ ] Fine-tune a pretrained **ResNet18** (ImageNet weights)
- [ ] Train a small **CNN from scratch** as a baseline for comparison
- [ ] Evaluate with **accuracy, macro-F1 and a confusion matrix**
- [ ] Scale up to the full dataset (2,000 / 160 / 350 per language)
- [ ] Extensions (planned):
  - **Grad-CAM:** which parts of the spectrogram the CNN focuses on
  - **Language timeline:** sliding 3-second windows over long audio to show which language is spoken when
  - **t-SNE language map:** do related languages cluster together?
  - **Accuracy vs clip length:** 0.5 s, 1 s, 2 s, 3 s
  - **Live demo** with Gradio on Hugging Face Spaces

---

## 🧰 Tech Stack

- **Python**
- **Hugging Face `datasets`:** streaming the FLEURS dataset
- **librosa:** audio processing and mel spectrograms
- **NumPy:** array handling and storage
- **Matplotlib:** visualization
- **PyTorch / torchvision:** model training *(next step)*
- **Google Colab:** development environment

---

## 📄 Acknowledgements

- **FLEURS dataset:** Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech* (2022). [arXiv:2205.12446](https://arxiv.org/abs/2205.12446)
