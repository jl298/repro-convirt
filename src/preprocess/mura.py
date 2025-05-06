import os
import numpy as np
import torch

from common.dataset import ClassificationDataset

def preprocess_mura(logger, data_path, train_transform, val_transform, percent=100, batch_size=32):
    # MURA dataset
    # Data source: https://stanfordmlgroup.github.io/competitions/mura/
    # Binary classification (normal vs abnormal)
    base_path = data_path
    num_classes = 1

    train_dir = os.path.join(base_path, "train")
    train_csv_path = os.path.join(base_path, "train_labeled_studies.csv")

    # Use validation set as test set as per the paper
    test_dir = os.path.join(base_path, "valid")
    test_csv_path = os.path.join(base_path, "valid_labeled_studies.csv")

    def load_study_labels(csv_path):
        study_labels = {}
        with open(csv_path, 'r') as f:
            for line in f:
                path, label = line.strip().split(',')
                study_labels[path] = int(label)
        return study_labels

    train_study_labels = load_study_labels(train_csv_path)
    test_study_labels = load_study_labels(test_csv_path)

    def load_image_paths_and_labels(base_dir, study_labels, data_dir):
        image_paths = []
        labels = []

        body_parts = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND',
                      'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

        for body_part in body_parts:
            body_part_dir = os.path.join(base_dir, body_part)
            if not os.path.exists(body_part_dir):
                continue

            patient_dirs = [os.path.join(body_part_dir, d) for d in os.listdir(body_part_dir)
                            if os.path.isdir(os.path.join(body_part_dir, d))]

            for patient_dir in patient_dirs:
                study_dirs = [os.path.join(patient_dir, d) for d in os.listdir(patient_dir)
                              if os.path.isdir(os.path.join(patient_dir, d))]

                for study_dir in study_dirs:
                    rel_path = os.path.relpath(study_dir, data_dir).replace('\\', '/')

                    if f"MURA-v1.1/{rel_path}/" in study_labels:
                        label = study_labels[f"MURA-v1.1/{rel_path}/"]
                    else:
                        continue

                    image_files = [os.path.join(study_dir, f) for f in os.listdir(study_dir)
                                   if f.endswith('.png')]

                    image_paths.extend(image_files)
                    labels.extend([label] * len(image_files))

        return image_paths, labels

    train_image_paths, train_labels = load_image_paths_and_labels(train_dir, train_study_labels, base_path)
    test_image_paths, test_labels = load_image_paths_and_labels(test_dir, test_study_labels, base_path)

    if percent < 100:
        train_data = list(zip(train_image_paths, train_labels))

        normal_data = [d for d in train_data if d[1] == 0]
        abnormal_data = [d for d in train_data if d[1] == 1]

        num_normal = max(1, int(len(normal_data) * percent / 100))
        num_abnormal = max(1, int(len(abnormal_data) * percent / 100))

        sampled_normal = np.random.choice(len(normal_data), size=num_normal, replace=False)
        sampled_abnormal = np.random.choice(len(abnormal_data), size=num_abnormal, replace=False)

        sampled_data = [normal_data[i] for i in sampled_normal] + [abnormal_data[i] for i in sampled_abnormal]
        np.random.shuffle(sampled_data)

        train_image_paths, train_labels = zip(*sampled_data)
        train_image_paths = list(train_image_paths)
        train_labels = list(train_labels)

    # Split training data into train/val (80/20)
    train_data = list(zip(train_image_paths, train_labels))
    np.random.shuffle(train_data)

    split_idx = int(len(train_data) * 0.8)
    train_data, val_data = train_data[:split_idx], train_data[split_idx:]

    if len(train_data) > 0:
        train_image_paths, train_labels = zip(*train_data)
        train_image_paths = list(train_image_paths)
        train_labels = list(train_labels)
    else:
        train_image_paths, train_labels = [], []

    if len(val_data) > 0:
        val_image_paths, val_labels = zip(*val_data)
        val_image_paths = list(val_image_paths)
        val_labels = list(val_labels)
    else:
        val_image_paths, val_labels = [], []

    train_labels = torch.tensor(train_labels, dtype=torch.float)
    val_labels = torch.tensor(val_labels, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.float)

    train_dataset = ClassificationDataset(
        train_image_paths,
        train_labels,
        transform=train_transform
    )

    val_dataset = ClassificationDataset(
        val_image_paths,
        val_labels,
        transform=val_transform
    )

    test_dataset = ClassificationDataset(
        test_image_paths,
        test_labels,
        transform=val_transform
    )

    logger.info(f"MURA dataset loaded:")
    logger.info(f"  Training set: {len(train_dataset)} images")
    logger.info(f"  Validation set: {len(val_dataset)} images")
    logger.info(f"  Test set: {len(test_dataset)} images")

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'num_classes': num_classes
    }
