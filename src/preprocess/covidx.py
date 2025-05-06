import os
import numpy as np
import torch

from common.dataset import ClassificationDataset

def preprocess_covidx(logger, data_path, train_transform, val_transform, percent=100, batch_size=32):
    # COVIDx
    # Data source: https://github.com/lindawangg/COVID-Net
    # Multi-class classification task: COVID-19, non-COVID pneumonia, and normal
    num_classes = 3

    base_path = data_path
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')

    class_names = ['COVID-19', 'pneumonia', 'normal']
    class_indices = {'COVID-19': 0, 'pneumonia': 1, 'normal': 2}

    train_txt = os.path.join(base_path, 'train_COVIDx9A.txt')
    test_txt = os.path.join(base_path, 'test_COVIDx9A.txt')

    def load_from_covidx_txt(txt_file, img_dir):
        img_paths = []
        labels = []

        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        patient_id = parts[0]
                        img_filename = parts[1]
                        class_name = parts[2]

                        if class_name in class_indices:
                            img_path = os.path.join(img_dir, img_filename)

                            if os.path.exists(img_path):
                                img_paths.append(img_path)
                                labels.append(class_indices[class_name])

        return img_paths, labels

    train_img_paths, train_labels = load_from_covidx_txt(train_txt, train_dir)
    test_img_paths, test_labels = load_from_covidx_txt(test_txt, test_dir)

    logger.info(f"COVIDx dataset loaded: {len(train_img_paths)} training images, {len(test_img_paths)} test images")

    if percent < 100:
        train_data = list(zip(train_img_paths, train_labels))

        class_grouped_data = {}
        for path, label in train_data:
            if label not in class_grouped_data:
                class_grouped_data[label] = []
            class_grouped_data[label].append((path, label))

        sampled_data = []
        for label, data in class_grouped_data.items():
            sample_size = max(1, int(len(data) * percent / 100))
            indices = np.random.choice(len(data), size=sample_size, replace=False)
            sampled_data.extend([data[i] for i in indices])

        np.random.shuffle(sampled_data)

        train_img_paths, train_labels = zip(*sampled_data)
        train_img_paths = list(train_img_paths)
        train_labels = list(train_labels)

    # Split training data into train/val (80/20)
    train_data = list(zip(train_img_paths, train_labels))
    np.random.shuffle(train_data)

    split_idx = int(len(train_data) * 0.8)
    train_data, val_data = train_data[:split_idx], train_data[split_idx:]

    if len(train_data) > 0:
        train_img_paths, train_labels = zip(*train_data)
        train_img_paths = list(train_img_paths)
        train_labels = list(train_labels)
    else:
        train_img_paths, train_labels = [], []

    if len(val_data) > 0:
        val_img_paths, val_labels = zip(*val_data)
        val_img_paths = list(val_img_paths)
        val_labels = list(val_labels)
    else:
        val_img_paths, val_labels = [], []

    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = ClassificationDataset(
        train_img_paths,
        train_labels,
        transform=train_transform
    )

    val_dataset = ClassificationDataset(
        val_img_paths,
        val_labels,
        transform=val_transform
    )

    test_dataset = ClassificationDataset(
        test_img_paths,
        test_labels,
        transform=val_transform
    )

    logger.info(f"COVIDx dataset prepared with {percent}% training data:")
    logger.info(f"  Training set: {len(train_dataset)} images")
    logger.info(f"  Validation set: {len(val_dataset)} images")
    logger.info(f"  Test set: {len(test_dataset)} images")

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'num_classes': num_classes
    }

