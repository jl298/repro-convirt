import os
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from common.model import ConVIRTModel
from common.dataset import (
    MedicalImageTextPairDataset,
    get_transforms_pretrain,
    build_dataloaders
)
from utils.checkpoint import get_last_checkpoint_from_filename, save_checkpoint
from utils.loss_tracker import LossTracker
from utils.logger import get_logger
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--dataset_type', default='chest', type=str, choices=['chest', 'bone'])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--lambda_weight', default=0.75, type=float)
    parser.add_argument('--proj_dim', default=512, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_interval', default=5, type=int)
    parser.add_argument('--workers', default=4, type=int)

    return parser.parse_args()


def main():
    set_seed()
    args = parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device:{args.device}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    embeddings_dir = os.path.join(args.output_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)

    print(f"Log directory set to: {log_dir}")
    logger = get_logger(log_dir, 'training.log')
    logger.info(f"Arguments: {args}")

    if args.dataset_type == 'bone':
        logger.error("\"bone\" is coming soon...")
        return

    logger.info("Creating model...")
    model = ConVIRTModel(
        img_encoder_name="resnet50",
        txt_encoder_name="emilyalsentzer/Bio_ClinicalBERT",
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        lambda_weight=args.lambda_weight
    )

    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    use_amp = torch.cuda.is_available() and args.device.type == 'cuda'
    logger.info(f"Using AMP (Mixed Precision Training): {use_amp}")
    
    scaler = None
    if use_amp:
        scaler = GradScaler()
        logger.info("Gradient scaler initialized for AMP")

    start_epoch = 0
    checkpoint_to_load = args.checkpoint
    checkpoint_data = None

    if args.resume and not checkpoint_to_load:
        checkpoint_to_load = get_last_checkpoint_from_filename(logger, checkpoint_dir)

    if checkpoint_to_load:
        logger.info(f"Loading checkpoint from {checkpoint_to_load}")
        try:
            checkpoint_data = torch.load(checkpoint_to_load, map_location=args.device)
            model.load_state_dict(checkpoint_data['model_state_dict'])

            checkpoint_epoch = checkpoint_data.get('epoch', 0)
            logger.info(f"Found checkpoint from epoch {checkpoint_epoch}")

            if args.resume:
                start_epoch = checkpoint_epoch + 1
                logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}/{args.epochs}")
            else:
                logger.info(f"Loaded weights from epoch {checkpoint_epoch}")
        except FileNotFoundError:
            logger.error(f"Checkpoint file not found: {checkpoint_to_load}")
            if args.resume:
                logger.error("Unable to resume without valid checkpoint.")
                raise
        except RuntimeError as e:
            logger.error(f"Failed to load checkpoint: {e}")
            if args.resume:
                logger.error("Unable to resume with incompatible checkpoint.")
                raise

    logger.info("Setting up datasets...")
    train_transform, val_transform = get_transforms_pretrain()

    train_dataset = MedicalImageTextPairDataset(
        data_path=os.path.join(args.data_path),
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
        transform=train_transform,
        dataset_type=args.dataset_type,
        split="train"
    )

    val_dataset = MedicalImageTextPairDataset(
        data_path=os.path.join(args.data_path),
        tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
        transform=val_transform,
        dataset_type=args.dataset_type,
        split="val"
    )

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Val set: {len(val_dataset)} samples")

    train_loader = build_dataloaders(train_dataset, batch_size=args.batch_size, num_workers=args.workers)
    val_loader = build_dataloaders(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    if args.resume and checkpoint_data is not None:
        if 'optimizer_state_dict' in checkpoint_data:
            try:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.info("Optimizer state restored")
            except ValueError:
                logger.warning("Optimizer state incompatible with current configuration")

        if 'scheduler_state_dict' in checkpoint_data and scheduler is not None:
            try:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                logger.info("Scheduler state restored")
            except ValueError:
                logger.warning("Scheduler state incompatible with current configuration")

        if 'scaler_state_dict' in checkpoint_data and scaler is not None:
            try:
                scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
                logger.info("GradScaler state restored")
            except ValueError:
                logger.warning("GradScaler state incompatible with current configuration")

    logger.info("Starting training...")

    loss_tracker = LossTracker(log_dir, logger=logger, resume=args.resume)

    if args.resume and checkpoint_data is not None:
        checkpoint_epoch = checkpoint_data.get('epoch', 0)
        loss_tracker.sync_with_checkpoint(checkpoint_epoch)

        if 'val_loss' in checkpoint_data:
            best_val_loss = checkpoint_data['val_loss']
            loss_tracker.update_best_val_loss(best_val_loss)
            logger.info(f"Best validation loss from checkpoint: {best_val_loss}")

    remaining_epochs = args.epochs - start_epoch
    logger.info(f"Training for {remaining_epochs} epochs (from epoch {start_epoch + 1} to {args.epochs})")

    model.to(args.device)

    best_val_loss = float('inf')
    best_model_path = None

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        model.train()
        train_losses = []

        train_progress = tqdm(train_loader, desc="Training")
        for batch in train_progress:
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(**batch)
                    loss = outputs["loss"]
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                outputs = model(**batch)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())
            train_progress.set_postfix({"train_loss": f"{loss.item():.4f}"})

        model.eval()
        val_losses = []

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validation")
            for batch in val_progress:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                if use_amp:
                    with autocast():
                        outputs = model(**batch)
                        loss = outputs["loss"]
                else:
                    outputs = model(**batch)
                    loss = outputs["loss"]

                val_losses.append(loss.item())
                val_progress.set_postfix({"val_loss": f"{loss.item():.4f}"})

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        if scheduler is not None:
            scheduler.step(val_loss)

        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Log GradScaler scale if using AMP
        if scaler is not None:
            logger.info(f"GradScaler scale: {scaler.get_scale()}")

        curr_lr = optimizer.param_groups[0]['lr']
        loss_tracker.on_epoch_end(epoch, train_loss, val_loss)
        if curr_lr not in loss_tracker.lr_history:
            loss_tracker.lr_history.append(curr_lr)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            loss_tracker.best_val_loss = val_loss

        if (epoch + 1) % args.save_interval == 0 or is_best or (epoch + 1) == args.epochs:
            checkpoint_path = save_checkpoint(
                logger=logger,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss,
                output_dir=args.output_dir,
                is_best=is_best,
                scaler=scaler
            )

            if is_best:
                best_model_path = checkpoint_path

    if best_model_path is None:
        best_model_path = os.path.join(args.output_dir, 'checkpoints', 'checkpoint_latest.pt')

    logger.info(f"Training completed. Best model: {best_model_path}")

    loss_tracker.analyze_loss('loss_analysis.txt')
    logger.info(f"Loss analysis report written to 'loss_analysis.txt'")

    loss_tracker.plot_loss_histogram('loss_changes_histogram.png')
    logger.info(f"Loss histogram saved to 'loss_changes_histogram.png'")

    logger.info("Pretraining finished successfully")


if __name__ == '__main__':
    main()