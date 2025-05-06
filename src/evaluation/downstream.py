import json
import os

import numpy as np
import torch
from torch import optim, nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from common.dataset import load_classification_dataset
from common.model import ClassificationModel
from utils.metrics import AverageMeter, compute_auc, compute_accuracy
from utils.logger import log_hyperparameters, log_epoch_metrics, log_evaluation_results


def evaluate_classification(model, args, logger):
    logger.info(f"Starting {args.mode} evaluation for {args.eval_task} with {args.eval_percent}% of data")

    use_amp = hasattr(args, 'use_amp') and args.use_amp and torch.cuda.is_available()

    dataset = load_classification_dataset(
        logger,
        args.data_path,
        args.eval_task,
        args.eval_percent,
        args.mode
    )

    train_loader = dataset['train_loader']
    val_loader = dataset['val_loader']
    test_loader = dataset['test_loader']
    num_classes = dataset['num_classes']

    encoder = model.get_image_encoder()

    freeze_encoder = (args.mode == 'linear')
    classifier = ClassificationModel(
        encoder=encoder,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder
    ).to(args.device)

    if args.mode == 'finetune':
        for param in encoder.parameters():
            param.requires_grad = False

        train_classifier_head(classifier, train_loader, args.device, num_classes, args.eval_task, use_amp=use_amp)

        for param in encoder.parameters():
            param.requires_grad = True

    criterion = get_criterion(num_classes, args.eval_task)

    if args.eval_task == 'covidx':
        lr = 1e-3
    else:  # rsna, chexpert, mura
        lr = 1e-4

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=lr,
        weight_decay=1e-6
    )

    lr_scheduler = None
    if args.mode == 'finetune':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

    is_multilabel = (num_classes > 1 and args.eval_task == 'chexpert')
    is_binary = (num_classes == 1 or is_multilabel)
    metric_name = 'AUC' if is_binary else 'Accuracy'

    best_val_metric = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0

    log_hyperparameters(logger, args)

    # Create scaler for AMP
    scaler = GradScaler() if use_amp else None

    for epoch in tqdm(range(100), desc=f"Training epochs ({args.mode})"):
        train_loss = train_epoch(
            classifier, train_loader, optimizer, criterion,
            args.device, num_classes, is_binary, args.eval_task,
            scaler=scaler, use_amp=use_amp
        )

        val_loss, val_metric, val_outputs, val_targets = evaluate_epoch(
            classifier, val_loader, criterion,
            args.device, num_classes, is_binary, args.eval_task,
            use_amp=use_amp
        )

        log_epoch_metrics(logger, epoch, train_loss, val_loss, val_metric, metric_name)

        if lr_scheduler:
            lr_scheduler.step(val_metric)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch + 1
            patience_counter = 0

            logger.info(f"New best model found at epoch {epoch + 1} with {metric_name}: {val_metric:.4f}")
            save_best_model(args, classifier, args.eval_task, args.mode, args.eval_percent)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch + 1}, no improvement for {patience} epochs')
                break

    logger.info(f'Best validation {metric_name}: {best_val_metric:.4f} at epoch {best_epoch}')

    best_model_path = get_best_model_path(args, args.eval_task, args.mode, args.eval_percent)
    classifier.load_state_dict(torch.load(best_model_path, map_location=args.device))

    _, test_metric, all_outputs, all_targets = evaluate_epoch(
        classifier, test_loader, criterion,
        args.device, num_classes, is_binary, args.eval_task,
        return_predictions=True, use_amp=use_amp
    )

    logger.info(f'Test {metric_name}: {test_metric:.4f}')

    outputs_dict = create_outputs_dict(
        args, args.mode, test_metric, metric_name, best_val_metric,
        best_epoch, all_outputs, all_targets, num_classes,
        args.eval_task
    )

    outputs_dict['embeddings'], outputs_dict['labels'] = extract_embeddings(
        classifier, test_loader, args.device, encoder, use_amp=use_amp
    )

    log_evaluation_results(logger, test_metric, metric_name, best_val_metric, best_epoch)

    return test_metric, outputs_dict, classifier, dataset


def train_classifier_head(classifier, train_loader, device, num_classes, task_name, max_steps=200, use_amp=False):
    criterion = get_criterion(num_classes, task_name)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=1e-3,
        weight_decay=1e-6
    )

    scaler = GradScaler() if use_amp else None

    classifier.train()
    train_loss = AverageMeter('Train Loss')
    step = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp):
            outputs = classifier(images)

            if num_classes == 1 or task_name == 'chexpert':
                if num_classes == 1:
                    outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels)

        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss.update(loss.item(), images.size(0))
        step += 1

        if step >= max_steps:
            break

    print(f"Classifier head trained. Final loss: {train_loss.avg:.4f}")


def get_criterion(num_classes, task_name):
    if num_classes > 1 and task_name != 'chexpert':
        return nn.CrossEntropyLoss()
    else:
        return nn.BCEWithLogitsLoss()


def train_epoch(model, dataloader, optimizer, criterion, device, num_classes, is_binary, task_name, scaler=None,
                use_amp=False):
    model.train()
    train_loss = AverageMeter('Train Loss')

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with autocast(enabled=use_amp):
            outputs = model(images)

            if is_binary:
                if num_classes == 1:
                    outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_loss.update(loss.item(), images.size(0))

    return train_loss.avg


def evaluate_epoch(model, dataloader, criterion, device, num_classes, is_binary, task_name, return_predictions=False,
                   use_amp=False):
    model.eval()
    val_loss = AverageMeter('Val Loss')
    all_outputs = []
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast(enabled=use_amp):
                outputs = model(images)

                if is_binary:
                    if num_classes == 1:
                        outputs = outputs.squeeze()
                    loss = criterion(outputs, labels.float())
                else:
                    loss = criterion(outputs, labels)

            val_loss.update(loss.item(), images.size(0))

            if use_amp:
                outputs = outputs.float()

            all_outputs.append(outputs.cpu())
            all_targets.append(labels.cpu())

            if return_predictions:
                if is_binary:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if is_binary:
        metric = compute_auc(all_outputs, all_targets, num_classes=num_classes)
    else:
        metric = compute_accuracy(all_outputs, all_targets)

    if return_predictions:
        all_preds = torch.cat(all_preds, dim=0)
        all_probs = torch.cat(all_probs, dim=0)
        return val_loss.avg, metric, (all_outputs, all_preds, all_probs), all_targets

    return val_loss.avg, metric, all_outputs, all_targets


def get_best_model_path(args, task_name, mode, percent):
    if os.path.basename(args.output_dir) == task_name:
        task_output_dir = args.output_dir
    else:
        task_output_dir = os.path.join(args.output_dir, task_name)

    return os.path.join(task_output_dir, f'best_{mode}_{task_name}_{percent}.pth')


def save_best_model(args, model, task_name, mode, percent):
    if os.path.basename(args.output_dir) == task_name:
        task_output_dir = args.output_dir
    else:
        task_output_dir = os.path.join(args.output_dir, task_name)

    os.makedirs(task_output_dir, exist_ok=True)
    best_model_path = os.path.join(task_output_dir, f'best_{mode}_{task_name}_{percent}.pth')
    torch.save(model.state_dict(), best_model_path)


def create_outputs_dict(args, mode, test_metric, metric_name, best_val_metric, best_epoch,
                        model_outputs, targets, num_classes, task_name):
    all_outputs, all_preds, all_probs = model_outputs
    all_targets = targets

    if num_classes == 1 or task_name == 'chexpert':
        if len(all_probs.shape) > 1 and all_probs.shape[1] > 1:
            y_score = all_probs[:, 1].unsqueeze(1)
        else:
            y_score = all_probs
    else:
        y_score = all_probs

    class_names = get_class_names(task_name, num_classes)

    results = {
        'task': args.eval_task,
        'mode': mode,
        'percent': args.eval_percent,
        'test_metric': float(test_metric),
        'metric_name': metric_name,
        'best_val_metric': float(best_val_metric),
        'best_epoch': best_epoch
    }

    if os.path.basename(args.output_dir) == args.eval_task:
        task_output_dir = args.output_dir
    else:
        task_output_dir = os.path.join(args.output_dir, args.eval_task)

    results_path = os.path.join(task_output_dir, f'results_{mode}_{args.eval_task}_{args.eval_percent}.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f)

    outputs_dict = {
        'y_true': all_targets.numpy(),
        'y_pred': all_preds.numpy(),
        'y_score': y_score.numpy(),
        'class_names': class_names
    }

    return outputs_dict


def get_class_names(task_name, num_classes):
    if task_name == 'rsna':
        return ['Normal', 'Pneumonia']
    elif task_name == 'chexpert':
        return ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    elif task_name == 'covidx':
        return ['COVID-19', 'Pneumonia', 'Normal']
    elif task_name == 'mura':
        return ['Normal', 'Abnormal']
    else:
        return [f'Class_{i}' for i in range(num_classes)]


def extract_embeddings(model, dataloader, device, encoder=None, use_amp=False):
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)

            with autocast(enabled=use_amp):
                features = encoder(images) if encoder else model.encoder(images)

            # Flatten features to 2D if they are 4D (CNN output)
            if len(features.shape) == 4:
                features = features.view(features.size(0), -1)  # Flatten to [batch_size, features]
            elif len(features.shape) == 3:
                features = features.mean(dim=1)  # Average across sequence dimension if needed

            if use_amp:
                features = features.float()

            embeddings.append(features.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)
