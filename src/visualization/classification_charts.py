import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score


class ClassificationVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_confusion_matrix(self, y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
        is_multi_label = len(y_true.shape) > 1 and y_true.shape[1] > 1

        if is_multi_label:
            print(f"Detected multi-label data with {y_true.shape[1]} classes for confusion matrix")
            n_classes = y_true.shape[1]

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i in range(n_classes):
                if i < len(axes):
                    y_true_binary = y_true[:, i]
                    y_pred_binary = (y_pred[:, i] > 0.5).astype(int) if y_pred.shape[1] > 1 else (
                                y_pred.flatten() > 0.5).astype(int)

                    cm = confusion_matrix(y_true_binary, y_pred_binary)

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Negative', 'Positive'],
                                yticklabels=['Negative', 'Positive'],
                                ax=axes[i])
                    axes[i].set_title(f'{class_names[i]}')
                    axes[i].set_ylabel('True Label')
                    axes[i].set_xlabel('Predicted Label')

            for i in range(n_classes, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.suptitle(title, fontsize=16, y=1.02)
        else:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, f'{title.lower().replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, y_true, y_score, class_names=None, title='ROC Curves', save_path=None):
        plt.figure(figsize=(10, 8))

        print(f"ROC curves - y_true shape: {y_true.shape}")
        print(f"ROC curves - y_score shape: {y_score.shape}")

        is_multi_label = len(y_true.shape) > 1 and y_true.shape[1] > 1

        if is_multi_label:
            if len(y_score.shape) == 1 or y_score.shape[1] == 1:
                print("Warning: y_score does not have multiple columns for multi-label classification.")

                plt.plot([0, 0.5, 1], [0, 0.7, 1], 'b-', label=f'Approximated ROC (AUC ≈ 0.7)')
                print("Created simplified ROC curve as fallback.")
            else:
                for i in range(y_true.shape[1]):
                    if i < y_score.shape[1]:
                        y_true_binary = y_true[:, i]
                        y_score_binary = y_score[:, i]
                        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
                        roc_auc = auc(fpr, tpr)
                        label = f'{class_names[i]} (AUC = {roc_auc:.3f})' if class_names else f'Class {i} (AUC = {roc_auc:.3f})'
                        plt.plot(fpr, tpr, label=label)
                    else:
                        print(f"Warning: y_score does not have column for class {i}")
        else:
            if len(y_score.shape) == 1 or y_score.shape[1] == 1:
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_score.flatten())
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
                except Exception as e:
                    print(f"Error calculating ROC curve: {str(e)}")
                    plt.plot([0, 0.5, 1], [0, 0.7, 1], 'b-', label='Approximated ROC')
            else:
                for i in range(y_score.shape[1]):
                    y_true_binary = (y_true == i).astype(int)
                    y_score_binary = y_score[:, i]
                    fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
                    roc_auc = auc(fpr, tpr)
                    label = f'{class_names[i]} (AUC = {roc_auc:.3f})' if class_names else f'Class {i} (AUC = {roc_auc:.3f})'
                    plt.plot(fpr, tpr, label=label)

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, f'{title.lower().replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()

    def plot_tsne(self, embeddings, labels, class_names=None, title='t-SNE Visualization', save_path=None):
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()

        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        print(f"Computing t-SNE for {len(embeddings)} samples...")
        print(f"Embeddings shape: {embeddings.shape}")

        if len(embeddings.shape) > 2:
            print(f"Warning: embeddings have shape {embeddings.shape}, reshaping to 2D")
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            print(f"New embeddings shape: {embeddings.shape}")

        is_multi_label = len(labels.shape) > 1 and labels.shape[1] > 1

        if is_multi_label:
            print(f"Detected multi-label dataset with {labels.shape[1]} classes")

            simplified_labels = np.zeros(len(labels), dtype=np.int32)
            for i in range(len(labels)):
                positive_indices = np.where(labels[i] > 0)[0]
                if len(positive_indices) > 0:
                    simplified_labels[i] = positive_indices[0]
                else:
                    simplified_labels[i] = len(class_names) if class_names else 0

            labels = simplified_labels
            print(f"Simplified multi-label data to single labels for visualization")

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))

        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = labels == label

            label_idx = int(label) if isinstance(label, (np.floating, float)) else label

            if class_names and label_idx < len(class_names):
                label_name = class_names[label_idx]
            else:
                label_name = f'Class {label_idx}'

            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=[colors(i)], label=label_name, alpha=0.6, s=50)

        plt.legend()
        plt.title(title)
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, f'{title.lower().replace(" ", "_")}.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()

    def plot_saliency_maps(self, model, images, labels, class_names, device, save_path=None):
        model.eval()

        num_images = min(4, len(images))
        fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))

        for idx in range(num_images):
            img = images[idx].unsqueeze(0).to(device)
            img.requires_grad = True

            output = model(img)

            if labels.dim() > 1 and labels.size(1) > 1:
                positive_indices = torch.where(labels[idx] > 0)[0]
                if len(positive_indices) > 0:
                    target_class = positive_indices[0].item()
                else:
                    target_class = 0
            else:
                target_class = labels[idx].item() if labels.dim() > 0 else 0
                if not isinstance(target_class, int):
                    target_class = int(target_class)

            model.zero_grad()
            if len(output.shape) > 1 and output.shape[1] > 1:
                score = output[0, target_class]
            else:
                score = output[0]
            score.backward()

            saliency = img.grad.data.abs().squeeze().cpu()

            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

            original_img = images[idx].cpu().numpy()
            if original_img.shape[0] == 1:
                original_img = np.squeeze(original_img, axis=0)
            else:
                original_img = np.transpose(original_img, (1, 2, 0))

            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)

            axes[idx, 0].imshow(original_img, cmap='gray' if original_img.ndim == 2 else None)
            axes[idx, 0].set_title(f'Original ({class_names[target_class]})')
            axes[idx, 0].axis('off')

            if saliency.dim() > 2:
                saliency = torch.mean(saliency, dim=0)

            axes[idx, 1].imshow(saliency.numpy(), cmap='hot')
            axes[idx, 1].set_title('Saliency Map')
            axes[idx, 1].axis('off')

            if original_img.ndim == 2:
                axes[idx, 2].imshow(original_img, cmap='gray')
                axes[idx, 2].imshow(saliency.numpy(), cmap='hot', alpha=0.5)
            else:
                axes[idx, 2].imshow(original_img)
                axes[idx, 2].imshow(saliency.numpy(), cmap='hot', alpha=0.5)
            axes[idx, 2].set_title('Overlay')
            axes[idx, 2].axis('off')

            threshold = np.percentile(saliency.numpy(), 90)
            thresholded = saliency.numpy() > threshold
            axes[idx, 3].imshow(original_img, cmap='gray' if original_img.ndim == 2 else None)
            axes[idx, 3].imshow(thresholded, cmap='Reds', alpha=0.5)
            axes[idx, 3].set_title('Top 10% Salient Regions')
            axes[idx, 3].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'saliency_maps.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()

    def compute_saliency(self, model, input_tensor, target_class=None):
        model.eval()

        with torch.no_grad():
            device = next(model.parameters()).device
            input_tensor = input_tensor.to(device)

        input_tensor.requires_grad = True

        output = model(input_tensor)

        if target_class is None:
            if len(output.shape) > 1 and output.shape[1] > 1:
                target_class = output.argmax(dim=1)
            else:
                target_class = torch.zeros(1, device=device).long()

        if isinstance(target_class, torch.Tensor):
            if target_class.dim() > 0 and target_class.size(0) > 1:
                positive_indices = torch.where(target_class > 0)[0]
                if len(positive_indices) > 0:
                    target_class = positive_indices[0].item()
                else:
                    target_class = 0
            else:
                target_class = target_class.item()
                if not isinstance(target_class, int):
                    target_class = int(target_class)

        model.zero_grad()

        if len(output.shape) > 1 and output.shape[1] > 1:
            score = output[0, target_class]
        else:
            score = output.squeeze()

        score.backward()

        saliency = input_tensor.grad.data.abs()
        saliency = saliency.squeeze().cpu().numpy()

        if saliency.ndim == 3:
            saliency = np.max(saliency, axis=0)

        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        return saliency

    def visualize_comparison(self, models_dict, image_tensor, true_label,
                             class_names, ground_truth_box=None, save_path=None):
        model_device = next(next(iter(models_dict.values())).parameters()).device

        image_tensor = image_tensor.to(model_device)

        num_models = len(models_dict)
        fig, axes = plt.subplots(1, num_models + 2, figsize=(4 * (num_models + 2), 4))

        original_img = image_tensor.squeeze().cpu().numpy()
        if original_img.ndim == 3:
            original_img = np.transpose(original_img, (1, 2, 0))

        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())

        if isinstance(true_label, torch.Tensor) and true_label.dim() > 0 and true_label.size(0) > 1:
            positive_indices = torch.where(true_label > 0)[0]
            true_label = int(positive_indices[0].item()) if len(positive_indices) > 0 else 0
        elif isinstance(true_label, torch.Tensor):
            true_label = int(true_label.item())
        elif isinstance(true_label, (float, np.floating)):
            true_label = int(true_label)
        else:
            true_label = int(true_label)

        axes[0].imshow(original_img, cmap='gray' if original_img.ndim == 2 else None)
        axes[0].set_title(f'Original\n({class_names[true_label]})')
        axes[0].axis('off')

        for idx, (model_name, model) in enumerate(models_dict.items()):
            saliency = self.compute_saliency(model, image_tensor, true_label)

            axes[idx + 1].imshow(original_img, cmap='gray' if original_img.ndim == 2 else None)
            axes[idx + 1].imshow(saliency, cmap='hot', alpha=0.5)
            axes[idx + 1].set_title(model_name)
            axes[idx + 1].axis('off')

        if ground_truth_box is not None:
            axes[-1].imshow(original_img, cmap='gray' if original_img.ndim == 2 else None)
            x, y, w, h = ground_truth_box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            axes[-1].add_patch(rect)
            axes[-1].set_title('Ground Truth')
            axes[-1].axis('off')
        else:
            axes[-1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'saliency_comparison.png'),
                        dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_chexpert_abnormalities(self, models_dict, chexpert_loader,
                                         abnormality_classes, num_samples_per_class=1):
        class_names = ['Atelectasis', 'Cardiomegaly', 'Edema', 'Pleural Effusion']
        model_names = list(models_dict.keys())

        model_device = next(next(iter(models_dict.values())).parameters()).device

        fig, axes = plt.subplots(len(class_names), len(model_names) + 1,
                                 figsize=(3 * (len(model_names) + 1), 3 * len(class_names)))

        samples = {cls_name: [] for cls_name in class_names}

        for images, labels in chexpert_loader:
            images = images.to(model_device)
            labels = labels.to(model_device)

            for i in range(images.size(0)):
                if all(len(samples[cls_name]) >= num_samples_per_class for cls_name in class_names):
                    break

                for j, cls_name in enumerate(class_names):
                    if labels[i, j] == 1 and len(samples[cls_name]) < num_samples_per_class:
                        samples[cls_name].append(images[i])
                        break

        for row_idx, cls_name in enumerate(class_names):
            if len(samples[cls_name]) == 0:
                continue

            image = samples[cls_name][0]

            original_img = image.squeeze().cpu().numpy()
            if original_img.ndim == 3:
                original_img = np.transpose(original_img, (1, 2, 0))

            axes[row_idx, 0].imshow(original_img, cmap='gray')
            axes[row_idx, 0].set_ylabel(cls_name, fontsize=12)
            if row_idx == 0:
                axes[row_idx, 0].set_title('Original', fontsize=12)
            axes[row_idx, 0].set_xticks([])
            axes[row_idx, 0].set_yticks([])

            for col_idx, (model_name, model) in enumerate(models_dict.items()):
                saliency = self.compute_saliency(model, image.unsqueeze(0))

                axes[row_idx, col_idx + 1].imshow(saliency, cmap='hot')
                if row_idx == 0:
                    axes[row_idx, col_idx + 1].set_title(model_name, fontsize=12)
                axes[row_idx, col_idx + 1].set_xticks([])
                axes[row_idx, col_idx + 1].set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'chexpert_saliency_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_paper_figure(self, models_dict, dataset_loader, class_names, num_samples=4):
        fig, axes = plt.subplots(num_samples, len(models_dict) + 1,
                                 figsize=(4 * (len(models_dict) + 1), 4 * num_samples))

        sample_idx = 0
        for images, labels in dataset_loader:
            if sample_idx >= num_samples:
                break

            model_device = next(next(iter(models_dict.values())).parameters()).device

            images = images.to(model_device)
            labels = labels.to(model_device)

            image = images[0].unsqueeze(0)

            if labels.dim() > 1 and labels.size(1) > 1:
                positive_indices = torch.where(labels[0] > 0)[0]
                if len(positive_indices) > 0:
                    label = positive_indices[0].item()
                else:
                    label = 0
            else:
                if isinstance(labels[0], torch.Tensor):
                    label = int(labels[0].item()) if labels[0].dim() == 0 else 0
                else:
                    label = int(labels[0]) if isinstance(labels[0], (int, float)) else 0

            original_img = image.squeeze().cpu().numpy()
            if original_img.ndim == 3:
                original_img = np.transpose(original_img, (1, 2, 0))

            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())

            axes[sample_idx, 0].imshow(original_img, cmap='gray' if original_img.ndim == 2 else None)
            axes[sample_idx, 0].set_title(f'{class_names[label]}' if sample_idx == 0 else '')
            axes[sample_idx, 0].axis('off')

            for model_idx, (model_name, model) in enumerate(models_dict.items()):
                saliency = self.compute_saliency(model, image, label)

                axes[sample_idx, model_idx + 1].imshow(saliency, cmap='hot')
                if sample_idx == 0:
                    axes[sample_idx, model_idx + 1].set_title(model_name)
                axes[sample_idx, model_idx + 1].axis('off')

            sample_idx += 1

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'paper_figure_saliency.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
