# Image Captioning Project

The project has two main parts:

- **`data_and_training.ipynb`** – This notebook prepares the data, builds the vocabulary, defines the model, and trains it. At the end, it saves the trained model and other files needed for inference.
- **`inference.ipynb`** – This notebook loads the trained model and uses it to generate captions for new images. It also shows some examples of good and bad captions using the BLEU score.

## 2. Model Architecture

The model uses an **Encoder-Decoder architecture with Attention**:

- **Encoder**: A pretrained ResNet-50 (CNN) extracts visual features from the image. Most of the ResNet layers are frozen; only the last few layers are fine-tuned during training.
- **Attention**: A Bahdanau-style attention mechanism helps the decoder focus on different parts of the image while generating each word.
- **Decoder**: An RNN-based decoder generates the caption word by word, using the attended image features and the previous words.

## 3. Dataset

The dataset contains:

- A folder of images (`Images/`)
- A file with captions (`captions.txt`), where each image has up to 5 human-written reference captions

## 4. Project Structure (Google Drive)

All data and results are stored in Google Drive, because the project runs on Google Colab. The folder is organized like this:

```
caption_data/
├── Images/                  # all training/validation/test images
├── captions.txt             # original human-written captions
└── trained/                 # everything produced by the training notebook
    ├── best_model.pth       # model checkpoint with the best validation loss
    ├── final_model.pth      # model checkpoint after the last training epoch
    ├── vocab.pkl            # vocabulary object (word <-> index mapping)
    ├── config.json          # model hyperparameters (embed_dim, hidden_dim, etc.)
    ├── splits.json          # train/val/test image splits
    ├── training_history.json
    ├── bleu_scores.json
    ├── loss_curves.png
    └── sample_captions_test.png
```

## 5. How the notebooks work

### `data_and_training.ipynb`
1. Mounts Google Drive and loads the images and captions.
2. Cleans the captions and builds a vocabulary.
3. Splits the data into train / validation / test sets.
4. Builds the Encoder-Decoder model with attention.
5. Trains the model and saves checkpoints, vocabulary, config, and splits to `trained/`.

### `inference.ipynb`
1. Mounts Google Drive and loads the config, vocabulary, and trained checkpoint.
2. Rebuilds the model architecture and loads the trained weights.
3. Generates captions for images, including images the model has never seen (test set).
4. Compares generated captions to human reference captions using the BLEU-2 score, and shows examples of successful and unsuccessful captions.

## 6. Requirements

The notebooks install their own dependencies, but the main libraries used are:

- `torch`, `torchvision`
- `pandas`, `numpy`
- `PIL` (Pillow)
- `matplotlib`
- `nltk` (for BLEU score)

---

## 7. Instructions for Running the Code

The full dataset and trained model are stored in our Google Drive folder, since these files are too large for GitHub. Please follow these steps to run `inference.ipynb` in Google Colab.

**Step 1 — Add the shared folder to your own Drive**
- Open this link: `https://drive.google.com/drive/u/4/folders/1U1uCQyef1X8NqZ7Ke_2T02VHTUPBzIFk`
- In Google Drive, right-click the `caption_data` folder → **"Organize" → "Add shortcut to Drive"**
- When asked where to put the shortcut, choose **"My Drive"** directly (not inside another folder). This is important, because the notebook expects this exact path.
- Check that the folder now appears at: `My Drive → caption_data`

**Step 2 — Open the notebook in Google Colab**
- Open `inference.ipynb` in Colab.

**Step 3 — Run all cells**
- Go to **Runtime → Run all**.
- When asked, click **"Connect to Google Drive"** and log in with your Google account to allow access.
- The notebook expects the folder at:
  ```
  /content/drive/MyDrive/caption_data
  ```
  If you placed the shortcut in a different location, either move it to the root of My Drive, or edit the `BASE_DIR` variable in the "Mount Google Drive & Paths" cell so it matches the correct path.

**Step 4 — Expected runtime**
- A GPU is not required for inference, but the notebook will use one automatically if it is available (**Runtime → Change runtime type → GPU**).
- Running all cells should take a few minutes.

**If you get a `FileNotFoundError`:**
This usually means the shortcut was not added to the correct location. Please check that the following file exists in your Drive:
```
/content/drive/MyDrive/caption_data/trained/best_model.pth
```
If it is in a different location, update the `BASE_DIR` variable in the notebook.
