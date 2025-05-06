import os
import numpy as np
import pandas as pd
import torch

from common.dataset import ClassificationDataset

def preprocess_chexpert(logger, data_path, train_transform, val_transform, percent=100, batch_size=32):
    # CheXpert - Multi-label binary classification task for 5 conditions
    # The 5 conditions are: Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion
    num_classes = 5
    target_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    base_path = data_path

    train_csv_path = os.path.join(base_path, 'train.csv')
    valid_csv_path = os.path.join(base_path, 'valid.csv')

    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)

    def process_path(path, base_dir):
        path = path.replace('\\', '/')
        if 'CheXpert-v1.0/' in path:
            parts = path.split('CheXpert-v1.0/')[1]
        else:
            parts = path
        return os.path.join(base_dir, parts)

    train_df['processed_path'] = train_df['Path'].apply(lambda p: process_path(p, base_path))
    valid_df['processed_path'] = valid_df['Path'].apply(lambda p: process_path(p, base_path))

    train_images = train_df['processed_path'].tolist()
    valid_images = valid_df['processed_path'].tolist()

    train_labels = train_df[target_labels].values.astype(np.float32)
    valid_labels = valid_df[target_labels].values.astype(np.float32)

    train_labels = np.nan_to_num(train_labels, nan=0.0)
    train_labels[train_labels == -1] = 0

    valid_labels = np.nan_to_num(valid_labels, nan=0.0)
    valid_labels[valid_labels == -1] = 0

    if percent < 100:
        num_samples = max(int(len(train_images) * percent / 100), 5 * len(target_labels))

        indices = np.random.choice(len(train_images), num_samples, replace=False)

        for label_idx, label in enumerate(target_labels):
            pos_indices = np.where(train_labels[:, label_idx] > 0)[0]
            if len(pos_indices) > 0 and not any(idx in indices for idx in pos_indices):
                if len(indices) > 0:
                    random_idx = np.random.choice(indices, 1)[0]
                    indices = np.delete(indices, np.where(indices == random_idx)[0])
                indices = np.append(indices, np.random.choice(pos_indices, 1))

        train_images = [train_images[i] for i in indices]
        train_labels = train_labels[indices]

    # Split valid set into validation and test sets (50/50)
    test_size = len(valid_images) // 2
    indices = np.random.permutation(len(valid_images))
    val_indices = indices[:test_size]
    test_indices = indices[test_size:]

    val_images = [valid_images[i] for i in val_indices]
    val_labels = valid_labels[val_indices]

    test_images = [valid_images[i] for i in test_indices]
    test_labels = valid_labels[test_indices]

    train_labels = torch.tensor(train_labels, dtype=torch.float)
    val_labels = torch.tensor(val_labels, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    train_dataset = ClassificationDataset(
        train_images,
        train_labels,
        transform=train_transform
    )

    val_dataset = ClassificationDataset(
        val_images,
        val_labels,
        transform=val_transform
    )

    test_dataset = ClassificationDataset(
        test_images,
        test_labels,
        transform=val_transform
    )

    logger.info(f"CheXpert dataset loaded with {percent}% training data:")
    logger.info(f"  Training set: {len(train_images)} images")
    logger.info(f"  Validation set: {len(val_images)} images")
    logger.info(f"  Test set: {len(test_images)} images")

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'num_classes': num_classes
    }

