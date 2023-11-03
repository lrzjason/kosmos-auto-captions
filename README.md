# Image Captioning with CLIP and Kosmos

This project is an image captioning application that uses the CLIP model from OpenAI and the Kosmos model from Microsoft. The application takes an image as input and generates a detailed description of the image, including any subject matter, style of art if any, and the context.

## Requirements

- Python 3.6 or later
- PyTorch 1.8.1 or later
- Transformers 4.35.0 or later
- diffusers[torch] 0.21.4 or later

## Installation

1. Clone this repository.
2. Install the required packages.

```bash
git clone https://github.com/lrzjason/kosmos-auto-captions.git
cd kosmos-auto-captions
pip install transformers==4.35.0 diffusers[torch]==0.21.4
```

# Hardware Requirement
\>= 12GB vram nvidia graphic card

# Usage
Set your input directory, output directory, and clip failed directory in the script.
Run the script with parameter:
input_dir: contains images which needs to caption
output_dir: output directory which would store the images and captions
clip_failed_dir: \[optional\] contains low scores(<15) captions and images

```bash
python autoCaptionsKosmos.py --input_dir /path/to/input --output_dir /path/to/output --clip_failed_dir /path/to/clip_failed
```

The script will process each image in the input directory, generate a caption for it, and save the caption to a text file in the output directory. If the CLIP score for the caption is below a certain threshold, the image and its caption will be saved to the clip failed directory instead.

# How It Works
The script uses the CLIP model to calculate a score for each generated caption. The score is a measure of how well the caption matches the image. If the score is below a certain threshold, the script considers the caption to be a failure and saves the image and its caption to a separate directory.

The script also uses the Kosmos model to generate the captions. The Kosmos model is a vision-to-sequence model that can generate a sequence of words from an image.

# Example Captions
![alt text](https://github.com/lrzjason/kosmos-auto-captions/blob/main/doc/American%20Bully%20-%20United%20State%20-%20Dog.jpg?raw=true)
The image features a black pit bull dog sitting on the grass, wearing a green collar. The dog is attentively looking at something in the distance, possibly a person or another dog. The scene takes place in a park, with a building in the background. There are several people in the scene, some standing closer to the dog and others further away. The people appear to be enjoying their time outdoors, possibly taking a break from their activities or simply enjoying the view.