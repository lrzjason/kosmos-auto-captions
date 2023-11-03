# Image Captioning with CLIP and Kosmos

This project is an image captioning application that uses the CLIP model from OpenAI and the Kosmos model from Microsoft. The application takes an image as input and generates a detailed description of the image, including any subject matter, style of art if any, and the context.

## Requirements

- Python 3.6 or later
- PyTorch 1.8.1 or later
- Transformers 4.10.0 or later
- Pillow 8.3.1 or later

## Installation

1. Clone this repository.
2. Install the required packages.

```bash
pip install torch==1.8.1 transformers==4.10.0 pillow==8.3.1
```

# Usage
Set your input directory, output directory, and clip failed directory in the script.
Run the script.

```bash
python script.py --input_dir /path/to/input --output_dir /path/to/output --clip_failed_dir /path/to/clip_failed
```

The script will process each image in the input directory, generate a caption for it, and save the caption to a text file in the output directory. If the CLIP score for the caption is below a certain threshold, the image and its caption will be saved to the clip failed directory instead.

# How It Works
The script uses the CLIP model to calculate a score for each generated caption. The score is a measure of how well the caption matches the image. If the score is below a certain threshold, the script considers the caption to be a failure and saves the image and its caption to a separate directory.

The script also uses the Kosmos model to generate the captions. The Kosmos model is a vision-to-sequence model that can generate a sequence of words from an image.