#!/bin/bash

# default path for the models directory relative to the script
models_dir="$(dirname "$(readlink -f "$0")")/models/original"

usage() {
  echo "Usage: $0 --dataset PATH [--output_dir PATH]"
  echo "You must pass a --dataset flag that points to a path of a saved dataset, such as --dataset ./clothes_dataset"
  echo "Optionally, you can specify an output directory for the finetuned model with --output_dir"
  exit 1
}

dataset_path=""
output_dir_set=false

# parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset) dataset_path="$2"; shift ;;
    --output_dir) output_dir="$2"; output_dir_set=true; shift ;;
    *) usage ;;
  esac
  shift
done

# check if dataset flag was passed in
if [ -z "$dataset_path" ]; then
  usage
fi

# check if the models/original directory exists which we will use as the model to finetune with
# in future maybe have an optional flag that points to model path for when we use quantized models?
if [ ! -d "$models_dir" ]; then
  echo "You must save the stable diffusion model to ./models/original"
  exit 1
fi

# set output_dir based on dataset_path if it was not already set from the output_dir flag
if [ "$output_dir_set" = false ]; then
  if [[ "$dataset_path" == *"clothes_dataset"* ]]; then
    output_dir="./models/clothes_finetuned_model"
  elif [[ "$dataset_path" == *"pixelart_dataset"* ]]; then
    output_dir="./models/pixelart_finetuned_model"
  else
    output_dir="./models/photograph_finetuned_model"
  fi
fi

accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$models_dir \
  --dataset_name=$dataset_path \
  --resolution=512 --center_crop --random_flip \
  --max_train_samples=3000 \
  --num_train_epochs=10 \
  --checkpointing_steps=1000 \
  --train_batch_size=16 \
  --enable_xformers_memory_efficient_attention \
  --output_dir=$output_dir \


# default train batch size is 16

# feel free to change any of the values for these flags