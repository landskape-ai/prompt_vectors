import os
import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser("Prompt Vectors for CLIP")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Which dataset to patch on for visual prompting.",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="padding",
        choices=["padding", "random_patch", "fixed_patch"],
        help="Which method to use for for visual prompting.",
    )
    parser.add_argument(
        "--prompt_size", type=int, default=30, help="Size of visual prompts.",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size of the images to be used for training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Number of examples to load in a mini batch.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers to use for data loading.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Number of steps to warmup for."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic results.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience.",
    )
    parser.add_argument(
        "--print_freq", type=int, default=10, help="Print frequency.",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Where to store the results, else does not store.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path of the checkpoint used for resume the training.",
    )
    parser.add_argument(
        "--evaluate",
        default=False,
        action="store_true",
        help="Whether to evaluate the model on test set?",
    )
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="Whether to use wandb for experiment tracking?",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Run name for the wandb experiment tracking.",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args
