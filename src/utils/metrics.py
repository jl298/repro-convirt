import os
import datetime
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

from utils.logger import save_metrics_to_json

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total


def compute_auc(outputs, targets, num_classes=1):
    probs = torch.sigmoid(outputs).cpu().numpy()
    targets = targets.cpu().numpy()

    if num_classes == 1:
        return roc_auc_score(targets, probs)
    else:
        return roc_auc_score(targets, probs, average='macro')


def compute_precision_at_k(similarity_matrix, query_labels, candidate_labels, k=10):
    num_queries = similarity_matrix.shape[0]
    precision_scores = []

    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()
    if isinstance(query_labels, torch.Tensor):
        query_labels = query_labels.cpu().numpy().tolist()
    elif not isinstance(query_labels, list):
        query_labels = query_labels.tolist() if hasattr(query_labels, 'tolist') else list(query_labels)
    if isinstance(candidate_labels, torch.Tensor):
        candidate_labels = candidate_labels.cpu().numpy().tolist()
    elif not isinstance(candidate_labels, list):
        candidate_labels = candidate_labels.tolist() if hasattr(candidate_labels, 'tolist') else list(candidate_labels)
    
    for i in range(num_queries):
        k_adjusted = min(k, len(candidate_labels))
        top_k_indices = np.argsort(-similarity_matrix[i])[:k_adjusted]

        retrieved_labels = [candidate_labels[idx] for idx in top_k_indices]
        query_label = query_labels[i]

        matches = sum(1 for label in retrieved_labels if label == query_label)
        precision = matches / k
        precision_scores.append(precision)

    return np.mean(precision_scores)

def compute_similarity_matrix(query_embeddings, candidate_embeddings):
    query_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    candidate_norm = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)

    query_embeddings_normalized = query_embeddings / np.maximum(query_norm, 1e-8)
    candidate_embeddings_normalized = candidate_embeddings / np.maximum(candidate_norm, 1e-8)

    similarity_matrix = np.matmul(query_embeddings_normalized, candidate_embeddings_normalized.T)

    return similarity_matrix


def extract_features(model, data_loader, device, is_text=False):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            if is_text:
                input_ids, attention_mask = batch[0], batch[1]
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                outputs = model.txt_encoder(input_ids=input_ids, attention_mask=attention_mask)

                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size())
                masked_embeddings = outputs.last_hidden_state * attention_mask_expanded
                masked_embeddings[attention_mask_expanded == 0] = -1e9
                features = torch.max(masked_embeddings, dim=1)[0]

                features = model.txt_projection(features)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            else:
                images = batch[0]
                images = images.to(device)

                features = model.img_encoder(images)

                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output if hasattr(features, 'pooler_output') else features

                if features.dim() > 2:
                    features = features.squeeze(-1).squeeze(-1)

                features = model.vis_projection(features)
                features = torch.nn.functional.normalize(features, p=2, dim=1)

            all_features.append(features.cpu())

            if len(batch) > 1:
                all_labels.append(batch[1].cpu())

    all_features = torch.cat(all_features, dim=0)

    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)
        return all_features, all_labels
    else:
        return all_features

def classification_result(args, metrics, outputs_dict):
    results_summary = {
        'task': args.task,
        'mode': args.mode,
        'percent': args.percent,
        'metric_value': float(metrics),
        'test_setup': {
            'device': str(args.device),
            'checkpoint': args.checkpoint,
            'output_dir': args.output_dir,
        }
    }

    save_metrics_to_json(args.log_dir, args.task, args.mode, args.percent, results_summary)

    log_file_path = os.path.join(args.log_dir, f"summary_{args.task}_{args.mode}_{args.percent}.txt")
    with open(log_file_path, 'w') as f:
        f.write(f"\n{'=' * 50}\n")
        f.write(f"Evaluation Summary for {args.task}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Data percentage: {args.percent}%\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Output directory: {args.output_dir}\n")
        f.write(f"Log directory: {args.log_dir}\n")
        f.write(f"Device: {args.device}\n\n")
        f.write(f"{'=' * 20} RESULTS {'=' * 20}\n\n")

        if args.mode == 'linear':
            f.write(f"Linear evaluation metric: {metrics:.4f}\n")
        else:
            f.write(f"Fine-tuning evaluation metric: {metrics:.4f}\n")

        if hasattr(outputs_dict, 'additional_metrics'):
            f.write("\nAdditional Metrics:\n")
            for k, v in outputs_dict['additional_metrics'].items():
                f.write(f"  {k}: {v}\n")

        if 'y_true' in outputs_dict and 'y_pred' in outputs_dict and 'class_names' in outputs_dict:
            try:
                y_true = outputs_dict['y_true']
                y_pred = outputs_dict['y_pred']

                if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                    y_true = np.argmax(y_true, axis=1)
                else:
                    y_true = y_true.ravel()

                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    y_pred = np.argmax(y_pred, axis=1)
                else:
                    y_pred = y_pred.ravel()

                cm = confusion_matrix(y_true, y_pred)
                class_names = outputs_dict['class_names']

                f.write(f"\n{'=' * 20} CONFUSION MATRIX {'=' * 20}\n\n")
                f.write(f"{'':^15}")
                for name in class_names:
                    f.write(f"{name:^15}")
                f.write("\n\n")

                for i, row in enumerate(cm):
                    f.write(f"{class_names[i]:^15}")
                    for val in row:
                        f.write(f"{val:^15d}")
                    f.write("\n")

                f.write("\n")

                from sklearn.metrics import precision_score, recall_score, f1_score

                if len(class_names) == 2:
                    precision = precision_score(y_true, y_pred, average='binary')
                    recall = recall_score(y_true, y_pred, average='binary')
                    f1 = f1_score(y_true, y_pred, average='binary')

                    f.write(f"\nPrecision: {precision:.4f}\n")
                    f.write(f"Recall: {recall:.4f}\n")
                    f.write(f"F1 Score: {f1:.4f}\n")
                else:
                    precision = precision_score(y_true, y_pred, average='weighted')
                    recall = recall_score(y_true, y_pred, average='weighted')
                    f1 = f1_score(y_true, y_pred, average='weighted')

                    f.write(f"\nWeighted Precision: {precision:.4f}\n")
                    f.write(f"Weighted Recall: {recall:.4f}\n")
                    f.write(f"Weighted F1 Score: {f1:.4f}\n")
            except Exception as e:
                f.write(f"\nError generating confusion matrix: {str(e)}\n")

        f.write(f"\n{'=' * 50}\n")
        f.write(f"Evaluation completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"Saved detailed log to: {log_file_path}")