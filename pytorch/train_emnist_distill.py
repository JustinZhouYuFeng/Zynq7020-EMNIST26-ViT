import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from tiny_vit_mnist import DATASET_CONFIGS, TinyViT, build_loaders, evaluate


class EmnistCnnTeacher(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_teacher_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    correct = 0
    total = 0
    loss_sum = 0.0
    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total += labels.numel()
        correct += int((logits.argmax(1) == labels).sum())
        loss_sum += float(loss) * labels.numel()
        if step % 100 == 0:
            print(
                f"teacher epoch {epoch} step {step:04d}: "
                f"loss={loss_sum / total:.4f}, acc={correct / total:.4f}"
            )
    return loss_sum / total, correct / total


def train_student_epoch(
    student,
    teacher,
    loader,
    optimizer,
    device,
    epoch,
    alpha,
    temperature,
    label_smoothing,
    focal_gamma,
):
    student.train()
    teacher.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)
        hard_loss_per_sample = F.cross_entropy(
            student_logits,
            labels,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        if focal_gamma > 0.0:
            true_class_prob = F.softmax(student_logits, dim=1).gather(
                1, labels.unsqueeze(1)
            ).squeeze(1)
            hard_loss_per_sample = (
                (1.0 - true_class_prob).pow(focal_gamma) * hard_loss_per_sample
            )
        hard_loss = hard_loss_per_sample.mean()
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)
        loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
        loss.backward()
        optimizer.step()
        total += labels.numel()
        correct += int((student_logits.argmax(1) == labels).sum())
        loss_sum += float(loss) * labels.numel()
        if step % 100 == 0:
            print(
                f"student epoch {epoch} step {step:04d}: "
                f"loss={loss_sum / total:.4f}, acc={correct / total:.4f}"
            )
    return loss_sum / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--teacher-out", default="checkpoints/emnist_cnn_teacher.pt")
    parser.add_argument("--teacher-in", default=None)
    parser.add_argument(
        "--student-in",
        default="checkpoints/tiny_vit_emnist_letters_depth3_mlp128_placcel.pt",
    )
    parser.add_argument(
        "--student-out",
        default="checkpoints/tiny_vit_emnist_letters_depth3_mlp128_distilled.pt",
    )
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--student-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--teacher-lr", type=float, default=8e-4)
    parser.add_argument("--student-lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--reuse-teacher", action="store_true")
    parser.add_argument("--finetune-teacher", action="store_true")
    parser.add_argument("--teacher-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DATASET_CONFIGS["emnist-letters"]
    train_loader, test_loader = build_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        "emnist-letters",
        augment=args.augment,
    )
    criterion = nn.CrossEntropyLoss()

    teacher = EmnistCnnTeacher(cfg["num_classes"]).to(device)
    os.makedirs(os.path.dirname(args.teacher_out), exist_ok=True)
    if args.reuse_teacher:
        teacher_load_path = args.teacher_in or args.teacher_out
        teacher_checkpoint = torch.load(teacher_load_path, map_location=device)
        teacher_best = float(teacher_checkpoint["test_acc"])
        teacher.load_state_dict(teacher_checkpoint["model"])
        if args.finetune_teacher and teacher_load_path != args.teacher_out:
            torch.save(teacher_checkpoint, args.teacher_out)
    else:
        teacher_best = 0.0

    if not args.reuse_teacher or args.finetune_teacher:
        teacher_optimizer = torch.optim.AdamW(
            teacher.parameters(), lr=args.teacher_lr, weight_decay=0.01
        )
        teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            teacher_optimizer, T_max=args.teacher_epochs
        )
        for epoch in range(1, args.teacher_epochs + 1):
            train_loss, train_acc = train_teacher_epoch(
                teacher, train_loader, teacher_optimizer, criterion, device, epoch
            )
            test_loss, test_acc = evaluate(teacher, test_loader, criterion, device)
            teacher_scheduler.step()
            print(
                f"teacher epoch {epoch}: train_loss={train_loss:.4f}, "
                f"train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, "
                f"test_acc={test_acc:.4f}"
            )
            if test_acc > teacher_best:
                teacher_best = test_acc
                torch.save(
                    {"model": teacher.state_dict(), "test_acc": test_acc},
                    args.teacher_out,
                )
        teacher_checkpoint = torch.load(args.teacher_out, map_location=device)
        teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    if args.teacher_only:
        print(f"teacher_best={teacher_best:.6f}")
        return

    student_checkpoint = torch.load(args.student_in, map_location=device)
    student = TinyViT(**student_checkpoint["config"]).to(device)
    student.load_state_dict(student_checkpoint["model"])
    student_best = float(student_checkpoint.get("test_acc") or 0.0)
    student_optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.student_lr, weight_decay=args.weight_decay
    )
    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        student_optimizer, T_max=args.student_epochs
    )
    os.makedirs(os.path.dirname(args.student_out), exist_ok=True)
    history = []
    for epoch in range(1, args.student_epochs + 1):
        train_loss, train_acc = train_student_epoch(
            student,
            teacher,
            train_loader,
            student_optimizer,
            device,
            epoch,
            args.alpha,
            args.temperature,
            args.label_smoothing,
            args.focal_gamma,
        )
        test_loss, test_acc = evaluate(student, test_loader, criterion, device)
        student_scheduler.step()
        print(
            f"student epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, "
            f"test_acc={test_acc:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": student_optimizer.param_groups[0]["lr"],
            }
        )
        if test_acc > student_best:
            student_best = test_acc
            output = dict(student_checkpoint)
            output["model"] = student.state_dict()
            output["test_acc"] = test_acc
            output["distillation"] = {
                "teacher": args.teacher_out,
                "teacher_test_acc": teacher_best,
                "alpha": args.alpha,
                "temperature": args.temperature,
                "label_smoothing": args.label_smoothing,
                "focal_gamma": args.focal_gamma,
                "augment": args.augment,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            }
            torch.save(output, args.student_out)

    history_path = os.path.splitext(args.student_out)[0] + "_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"teacher_best={teacher_best:.6f}")
    print(f"student_best={student_best:.6f}")


if __name__ == "__main__":
    main()
