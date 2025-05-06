import os
import time
import json
import matplotlib.pyplot as plt

class LossTracker:
    def __init__(self, log_dir, logger, resume=False):
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []
        self.log_dir = log_dir
        self.logger = logger
        self.best_val_loss = float('inf')
        self.start_time = time.time()
        self.current_epoch = 0

        if resume:
            loss_file = os.path.join(log_dir, 'loss_history.json')
            if os.path.exists(loss_file):
                try:
                    with open(loss_file, 'r') as f:
                        data = json.load(f)
                        self.train_losses = data.get('train_loss', [])
                        self.val_losses = data.get('val_loss', [])
                        self.lr_history = data.get('lr', [])
                        self.best_val_loss = data.get('best_val_loss', float('inf'))
                        self.current_epoch = len(self.train_losses)
                        self.logger.info(f"Loaded previous training history: {self.current_epoch} epochs")
                except:
                    self.logger.warning(f"Couldn't load loss history from {loss_file}. Starting fresh.")
    
    def update_best_val_loss(self, new_val_loss):
        self.best_val_loss = new_val_loss
        
    def sync_with_checkpoint(self, checkpoint_epoch):
        if checkpoint_epoch is None or checkpoint_epoch >= len(self.train_losses) - 1:
            return
            
        self.logger.info(f"Syncing with checkpoint: truncating from {len(self.train_losses)} to {checkpoint_epoch+1} epochs")
        self.train_losses = self.train_losses[:checkpoint_epoch+1]
        self.val_losses = self.val_losses[:checkpoint_epoch+1]

        if len(self.lr_history) > checkpoint_epoch+1:
            self.lr_history = self.lr_history[:checkpoint_epoch+1]
            
        self.current_epoch = len(self.train_losses)

    def on_epoch_end(self, epoch, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        elapsed_time = time.time() - self.start_time

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        loss_data = {
            'epochs': list(range(1, epoch + 2)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'lr': self.lr_history,
            'best_val_loss': self.best_val_loss,
            'elapsed_time': elapsed_time
        }

        with open(os.path.join(self.log_dir, 'loss_history.json'), 'w') as f:
            json.dump(loss_data, f, indent=2)

        self.plot_loss_line_chart(epoch + 1)

        if len(self.val_losses) > 10:
            recent_losses = self.val_losses[-10:]
            loss_diff = recent_losses[0] - recent_losses[-1]
            if loss_diff < 0.01:
                self.logger.info("Loss barely changing - probably converging soon!")

    def analyze_loss(self, filename):
        final_analysis_path = os.path.join(self.log_dir, filename)
        with open(final_analysis_path, 'w') as f:
            f.write("====== ConVIRT Training Loss Analysis ======\n\n")
            f.write(f"Total epochs: {len(self.train_losses)}\n")
            f.write(f"Best validation loss: {self.best_val_loss:.6f}\n")
            f.write(f"Final training loss: {self.train_losses[-1]:.6f}\n")
            f.write(f"Final validation loss: {self.val_losses[-1]:.6f}\n\n")

            if len(self.train_losses) > 10:
                train_loss_change = self.train_losses[0] - self.train_losses[-1]
                val_loss_change = self.val_losses[0] - self.val_losses[-1]

                f.write("=== Learning Curve Analysis ===\n")
                f.write(f"Initial train loss: {self.train_losses[0]:.6f}\n")
                f.write(f"Initial validation loss: {self.val_losses[0]:.6f}\n")
                f.write(f"Total train loss reduction: {train_loss_change:.6f} ({train_loss_change / self.train_losses[0] * 100:.2f}%)\n")
                f.write(f"Total validation loss reduction: {val_loss_change:.6f} ({val_loss_change / self.val_losses[0] * 100:.2f}%)\n\n")

                recent_train = self.train_losses[-10:]
                recent_val = self.val_losses[-10:]
                recent_train_change = recent_train[0] - recent_train[-1]
                recent_val_change = recent_val[0] - recent_val[-1]

                f.write("=== Recent Convergence Analysis (Last 10 Epochs) ===\n")
                f.write(f"Recent train loss change: {recent_train_change:.6f} ({recent_train_change / recent_train[0] * 100:.2f}%)\n")
                f.write(f"Recent validation loss change: {recent_val_change:.6f} ({recent_val_change / recent_val[0] * 100:.2f}%)\n")

                train_val_diff_start = abs(self.train_losses[0] - self.val_losses[0])
                train_val_diff_end = abs(self.train_losses[-1] - self.val_losses[-1])

                f.write("\n=== Overfitting Check ===\n")
                f.write(f"Initial train-val gap: {train_val_diff_start:.6f}\n")
                f.write(f"Final train-val gap: {train_val_diff_end:.6f}\n")

                if train_val_diff_end > 2 * train_val_diff_start:
                    f.write("Warning! Train-val gap has widened a lot. Might be overfitting. Try more dropout.\n")
                else:
                    f.write("No obvious overfitting detected.\n")

    def plot_loss_line_chart(self, epochs):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(range(1, epochs + 1), self.train_losses, 'b-', label='Train Loss')
        plt.plot(range(1, epochs + 1), self.val_losses, 'r-', label='Val Loss')
        plt.grid(True)
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        if len(self.train_losses) > 1:
            train_loss_diffs = [self.train_losses[i] - self.train_losses[i - 1] for i in
                                range(1, len(self.train_losses))]
            val_loss_diffs = [self.val_losses[i] - self.val_losses[i - 1] for i in range(1, len(self.val_losses))]

            plt.subplot(2, 1, 2)
            plt.plot(range(2, epochs + 1), train_loss_diffs, 'b--', label='Train Loss Δ')
            plt.plot(range(2, epochs + 1), val_loss_diffs, 'r--', label='Val Loss Δ')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.grid(True)
            plt.legend()
            plt.title('Loss Change per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Change')

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'loss_plot.png'))
        plt.close()

    def plot_loss_histogram(self, filename):
        if len(self.train_losses) > 5:
            plt.figure(figsize=(10, 6))

            train_diffs = [self.train_losses[i] - self.train_losses[i - 1] for i in
                           range(1, len(self.train_losses))]
            plt.hist(train_diffs, bins=20, alpha=0.5, label='Train Loss Changes')

            val_diffs = [self.val_losses[i] - self.val_losses[i - 1] for i in
                         range(1, len(self.val_losses))]
            plt.hist(val_diffs, bins=20, alpha=0.5, label='Val Loss Changes')

            plt.axvline(x=0, color='k', linestyle='--')
            plt.xlabel('Loss Change per Epoch')
            plt.ylabel('Frequency')
            plt.title('Distribution of Loss Changes')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.log_dir, filename))
            plt.close()
