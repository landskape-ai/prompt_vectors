from __future__ import print_function

import os
import sys
import time
import wandb
import random
from tqdm import tqdm

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

import clip
import src.prompters as prompters

from src.args import parse_arguments
from src.datasets.common import get_dataloader
from src.datasets.registry import get_dataset
from src.datasets.templates import get_templates

# from src.eval import evaluate
from src.utils import (
    cosine_lr,
    convert_models_to_fp32,
    refine_classname,
    accuracy,
    save_checkpoint,
    AverageMeter,
    ProgressMeter,
)


def finetune(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True

    dataset = args.dataset
    ckpdir = os.path.join(args.save, dataset)

    assert dataset is not None, "Please provide a training dataset."
    template_fn = get_templates(dataset)[0]
    args.template_str = template_fn("`<class>`")
    print(f"template: {args.template_str}")

    train_dataset = get_dataset(
        dataset, preprocess=preprocess, location=args.data, batch_size=args.batch_size,
    )
    val_dataset = get_dataset(
        dataset + "Val", preprocess=preprocess, location=args.data, batch_size=args.batch_size,
    )
    train_loader = get_dataloader(train_dataset, is_train=True, args=args, image_encoder=None)
    val_loader = get_dataloader(val_dataset, is_train=False, args=args, image_encoder=None)

    class_names = train_dataset.classes
    class_names = refine_classname(class_names)
    texts = [template_fn(label) for label in class_names]

    # create model
    model, preprocess = clip.load(args.model, args.device, jit=False)
    convert_models_to_fp32(model)
    model.eval()

    prompter = prompters.__dict__[args.method](args).to(args.device)

    # define criterion and optimizer
    optimizer = torch.optim.AdamW(prompter.parameters(), lr=args.lr, weight_decay=args.wd)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_steps, total_steps)

    # wandb
    if args.use_wandb:
        wandb.init(
            entity="adversarial-reprogramming-vqa",
            project="Prompt Vectors for CLIP",
            name=args.run_name,
            dir=args.save,
            save_code=True,
            config=vars(args),
        )
        wandb.watch(prompter, criterion, log="all", log_freq=10)

    epochs_since_improvement = 0
    best_acc1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to single gpu.
            checkpoint = torch.load(args.resume, map_location=args.device)
            args.start_epoch = checkpoint["epoch"]
            prompter.load_state_dict(checkpoint["state_dict"])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
            optimizer = checkpoint["optimizer"]
            scheduler = checkpoint["scheduler"]
            epochs_since_improvement = checkpoint["epochs_since_improvement"]
            best_acc1 = checkpoint["best_acc1"]
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        acc1 = validate(val_loader, texts, model, prompter, criterion, args)
        return

    for epoch in range(args.epochs):

        # train for one epoch
        train(
            train_loader,
            texts,
            model,
            prompter,
            optimizer,
            scheduler,
            criterion,
            scaler,
            epoch,
            args,
        )

        # evaluate on validation set
        acc1 = validate(val_loader, texts, model, prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

        save_checkpoint(
            {
                "args": args,
                "epoch": epoch + 1,
                "epochs_since_improvement": epochs_since_improvement,
                "best_acc1": best_acc1,
                "state_dict": prompter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            ckpdir,
            is_best=is_best,
        )

        if epochs_since_improvement >= args.patience:
            print("The training halted by early stopping criterion.")
            break

    if args.use_wandb:
        wandb.run.finish()


def train(
    train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(args.device)
        target = target.to(args.device)
        text_tokens = clip.tokenize(texts).to(args.device)

        # with automatic mixed precision
        with autocast():
            prompted_images = prompter(images)
            output, _ = model(prompted_images, text_tokens)
            loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({"train/loss": losses.avg, "train/acc": top1.avg})
    return losses.avg, top1.avg


@torch.no_grad()
def validate(val_loader, texts, model, prompter, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1_org = AverageMeter("Original Acc@1", ":6.2f")
    top1_prompt = AverageMeter("Prompt Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1_org, top1_prompt], prefix="Validate: ",
    )

    # switch to evaluation mode
    prompter.eval()

    end = time.time()
    for i, (images, target) in enumerate(tqdm(val_loader)):

        images = images.to(args.device)
        target = target.to(args.device)
        text_tokens = clip.tokenize(texts).to(args.device)
        prompted_images = prompter(images)

        # compute output
        output_prompt, _ = model(prompted_images, text_tokens)
        output_org, _ = model(images, text_tokens)
        loss = criterion(output_prompt, target)

        # measure accuracy and record loss
        acc1 = accuracy(output_prompt, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1_prompt.update(acc1[0].item(), images.size(0))

        acc1 = accuracy(output_org, target, topk=(1,))
        top1_org.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print(
        " * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}".format(
            top1_prompt=top1_prompt, top1_org=top1_org
        )
    )

    if args.use_wandb:
        wandb.log(
            {
                "val/loss": losses.avg,
                "val/acc_prompt": top1_prompt.avg,
                "val/acc_org": top1_org.avg,
            }
        )

    return top1_prompt.avg


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("No GPU available!")
        sys.exit(1)

    data = "data"
    models = ["ViT-B-32"]
    datasets = [
        "Cars",
        "DTD",
        "EuroSAT",
        "GTSRB",
        "MNIST",
        "RESISC45",
        "SUN397",
        "SVHN",
    ]
    epochs = {
        # 'Cars': 35,
        # 'DTD': 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4,
        # 'ImageNet': 4
    }

    for model in models:
        for dataset in datasets:
            print("-" * 100)
            print(f"Finetuning {model} on {dataset}")
            print("-" * 100)
            args = parse_arguments()
            torch.cuda.set_device(int(args.gpu))
            args.lr = 1e-5
            args.epochs = epochs[dataset]
            args.data = data
            args.dataset = dataset
            args.batch_size = 128
            args.model = model
            args.save = f"checkpoints/{model}"
            args.device = "cuda"
            finetune(args)
