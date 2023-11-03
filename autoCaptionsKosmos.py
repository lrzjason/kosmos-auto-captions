import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, CLIPProcessor, CLIPModel
import torch
import os
import shutil
import json
import argparse

seed = 1
CLIPSCORE_THRESHOLD = 15
INSTRUCT_PROMPT = 'Give a detailed description of this image, including any subject matter, style of art if any, and the context:'
# MAXIUMN_RETRY = 1

torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model_and_processor():
  if calc_clip:
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",device_map="cuda")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",device_map="cuda")
  model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224",device_map="cuda")
  processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224",device_map="cuda")
  return clip_processor, clip_model, model, processor

def setup_argparse():
    parser = argparse.ArgumentParser(description='Image Captioning with CLIP and Kosmos')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--clip_failed_dir', type=str, required=False, help='Directory for images with low CLIP scores')
    return parser.parse_args()

# input_directory = 'F:/ImageSet/dump/mobcup_output_empty'
# output_directory = 'F:/ImageSet/dump/mobcup_output_kosmos'
# clip_failed_directory = 'F:/ImageSet/dump/mobcup_output_kosmos_clip_failed'

def load_txt(content):
  data = []
  chunk_size = 77
  for i in range(0, len(content), chunk_size):
    chunk = content[i:i+chunk_size]
  #   prompt_tokens = clip.tokenize(chunk)
    data.append(chunk)
  return data

def calc_clip_score(text,img,clip_processor,clip_model):
  text_data = load_txt(text)
  inputs = clip_processor(text=text_data, images=img, return_tensors="pt", padding=True).to(device)
  outputs = clip_model(**inputs)
  score = outputs.logits_per_image.mean().item()
  return score
  

def write_text(filename,output_directory,content):
  output_path = os.path.join(output_directory, filename)
  # print('output_path: ', output_path)
  # create output_prompt_path file if not exists
  if not os.path.exists(output_path):
    open(output_path, 'a').close()
  with open(output_path, 'r+',encoding='utf-8') as output_file:
    output_file.truncate(0)
    output_file.write(content)

def caption(prompt,image,processor, model):
  inputs = processor(text=prompt, images=image, return_tensors="pt")
  inputs.to(device)
  generated_ids = model.generate(
      pixel_values=inputs["pixel_values"],
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      image_embeds=None,
      image_embeds_position_mask=inputs["image_embeds_position_mask"],
      use_cache=True,
      max_new_tokens=128,
  )
  generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

  # Specify `cleanup_and_extract=False` in order to see the raw model generation.
  # processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

  # print(processed_text)
  # `<grounding> An image of<phrase> a snowman</phrase><object><patch_index_0044><patch_index_0863></object> warming himself by<phrase> a fire</phrase><object><patch_index_0005><patch_index_0911></object>.`

  # By default, the generated  text is cleanup and the entities are extracted.
  processed_text, _ = processor.post_process_generation(generated_text)
  processed_text = processed_text.replace(prompt,'').strip()
  return processed_text

calc_clip = False

def main():
  args = setup_argparse()
  input_directory = args.input_dir
  output_directory = args.output_dir
  clip_failed_directory = args.clip_failed_dir
  if clip_failed_directory is not None:
     calc_clip = True
  clip_processor, clip_model, model, processor = setup_model_and_processor()

  # create output directory if not exists
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  # create output directory if not exists
  if not os.path.exists(clip_failed_directory):
    os.makedirs(clip_failed_directory)

  file_count = 0
  # prompt = "<grounding>Give a detailed description of this image, including any subject matter, style of art if any, and the context:</grounding>"
  prompt = INSTRUCT_PROMPT

  temp_dir = os.path.join(os.getcwd(), 'temp')
  if not os.path.exists(temp_dir):
      os.makedirs(temp_dir)
  result_file = os.path.join(temp_dir, 'result.json')
  if os.path.exists(result_file):
      os.remove(result_file)
      # create result file
  score_results = []
  clip_failed_results = []

  score_acc = 0

  # loop through all files in input directory
  for filename in os.listdir(input_directory):
    file_count+=1
    # re init model and tokenizer every 10 files
    # if file_count % RE_INIT_MODEL_WHILE_FILES == 0:
    #   print('--------------Re init model and tokenizer')
    #   init_model(file_count,model,tokenizer)
    print('--------------filename: ', filename)
    # skip if file is not jpg or png
    if not filename.lower().endswith('.jpg') and not filename.lower().endswith('.png'):
      continue

    # check if file is already exist in output directory
    # if exist, skip this file
    if os.path.exists(os.path.join(output_directory, filename)):
      print('--------------File already exist. Skip this file: ', filename)
      continue
    
    ori_filename = filename
    filename_without_extention = os.path.splitext(filename)[0]
    suffix = ''
    prompt_file = f'{filename_without_extention}{suffix}.txt'

    history = None
    image_path = os.path.join(input_directory, filename)
    print('--------------image_path: ', image_path)

    image = Image.open(image_path)

    clip_score = 0
    retry_count = 0
    processed_text = ''
    processed_text = caption(prompt,image,processor, model)
    print('processed_text: ', processed_text)
    write_text(prompt_file,output_directory,processed_text)

    # copy image to classified folder
    output_image_path = os.path.join(output_directory, filename)
    print('output_image_path: ', output_image_path)
    # copy image from image_path to output_image_path, overwrite if exists
    shutil.copyfile(image_path, output_image_path)
    # copy image to classified folder
    if calc_clip:
      # calc the clip score
      clip_score = calc_clip_score(processed_text,image,clip_processor,clip_model)
      print('clip_score: ', clip_score)
      # put image to clip failed folder if clip score is too low
      if clip_score < CLIPSCORE_THRESHOLD:
        clip_failed_results.append({'filename': ori_filename, 'score': clip_score})
        clip_failed_image_path = os.path.join(clip_failed_directory, filename)
        print('clip_failed_image_path: ', clip_failed_image_path)
        # copy image from image_path to output_image_path, overwrite if exists
        shutil.copyfile(image_path, clip_failed_image_path)
        # copy text to failed folder
        write_text(prompt_file,clip_failed_directory,processed_text)
      
      score_results.append({'filename': ori_filename, 'score': clip_score})
      score_acc += clip_score
      
      # Save the results to a JSON file
      with open(result_file, 'w') as f:
          json.dump(score_results, f)
      average_score = score_acc / len(score_results)
      score_results.append({'filename': 'average_score', 'score': average_score})
      print('score_acc / sample_num: ', average_score)

    print('--------------End: ', filename)

    # break
  print('Process Completed.')


if __name__ == "__main__":
    main()