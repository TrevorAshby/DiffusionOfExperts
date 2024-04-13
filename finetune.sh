#!/bin/bash

# default path for the model_downloads directory relative to the script
# models_dir="$(dirname "$(readlink -f "$0")")/model_downloads/original"
models_dir="CompVis/stable-diffusion-v1-4"

. .venv/bin/activate

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

# # check if the model_downloads/original directory exists which we will use as the model to finetune with
# # in future maybe have an optional flag that points to model path for when we use quantized models?
# if [ ! -d "$models_dir" ]; then
#   echo "You must save the stable diffusion model to ./model_downloads/original"
#   exit 1
# fi

# set output_dir based on dataset_path if it was not already set from the output_dir flag
if [ "$output_dir_set" = false ]; then
  if [[ "$dataset_path" == *"clothes_dataset"* ]]; then
    output_dir="./model_downloads/clothes_finetuned_model"
  elif [[ "$dataset_path" == *"pixelart_dataset"* ]]; then
    output_dir="./model_downloads/pixelart_finetuned_model"
  else
    output_dir="./model_downloads/photograph_finetuned_model"
  fi
fi

# --num_processes=1 --main_process_port=29501 --gpu_ids=1
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$models_dir \
  --dataset_name=$dataset_path \
  --learning_rate=1e-05 \
  --max_train_samples=1000 \
  --num_train_epochs=20 \
  --checkpointing_steps=500 \
  --train_batch_size=4 \
  --resume_from_checkpoint=latest \
  --output_dir=$output_dir \


# default train batch size is 16

# feel free to change any of the values for these flags