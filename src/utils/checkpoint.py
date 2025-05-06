import os
import shutil
import torch
import glob

def get_last_checkpoint_from_filename(logger, checkpoint_dir):
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    if os.path.exists(latest_checkpoint_path):
        logger.info(f"Resume mode: Using latest checkpoint at {latest_checkpoint_path}")
        return latest_checkpoint_path
    else:
        logger.warning("No checkpoint_latest.pt found. Looking for other checkpoints...")
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
        if checkpoint_files:
            checkpoint_epochs = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in checkpoint_files]
            latest_epoch = max(checkpoint_epochs)
            checkpoint_to_load = os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pt')
            logger.info(f"Resume mode: Found latest checkpoint at epoch {latest_epoch}")
            return checkpoint_to_load

    return None

def save_checkpoint(logger, epoch, model, optimizer, scheduler, val_loss, output_dir, is_best=False, save_interval=5, scaler=None):
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)

    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    shutil.copy(checkpoint_path, latest_path)
    logger.info(f"Saved latest checkpoint to {latest_path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        shutil.copy(checkpoint_path, best_path)
        logger.info(f"Saved best model checkpoint to {best_path}")

    cleanup_old_checkpoints(logger, checkpoint_dir, epoch, keep_every=save_interval)

    return checkpoint_path

def cleanup_old_checkpoints(logger, checkpoint_dir, current_epoch, keep_every=5):
    keep_files = [
        os.path.join(checkpoint_dir, 'checkpoint_best.pt'),
        os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    ]

    for e in range(0, current_epoch + 1, keep_every):
        keep_files.append(os.path.join(checkpoint_dir, f'checkpoint_epoch_{e}.pt'))

    for checkpoint_file in glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')):
        if checkpoint_file not in keep_files:
            try:
                os.remove(checkpoint_file)
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {checkpoint_file}: {e}")