import os
import pandas as pd
from sklearn.model_selection import train_test_split

from common.dataset import ClassificationDataset

def preprocess_rsna(logger, data_path, train_transform, val_transform, percent=100, batch_size=32):

    base_path = data_path
    num_classes = 2  # normal(0), pneumonia(1) - Binary classfication

    train_images_dir = os.path.join(base_path, 'stage_2_train_images')
    test_images_dir = os.path.join(base_path, 'stage_2_test_images')
    labels_file = os.path.join(base_path, 'stage_2_train_labels.csv')

    df = pd.read_csv(labels_file)

    # Group by patientId to process multiple bounding boxes of the same patient
    # Consider it as pneumonia, if even one case where Target is 1
    patient_df = df.groupby('patientId')['Target'].max().reset_index()

    patient_df['image_path'] = patient_df['patientId'].apply(
        lambda x: os.path.join(train_images_dir, f'{x}.dcm')
    )

    patient_df['label'] = patient_df['Target']

    if percent < 100:
        train_df = pd.DataFrame()
        for label in range(num_classes):
            class_df = patient_df[patient_df['label'] == label]
            sample_size = max(1, int(len(class_df) * percent / 100))  # 1 or more
            sampled_df = class_df.sample(n=sample_size, random_state=42)
            train_df = pd.concat([train_df, sampled_df])
    else:
        train_df = patient_df

    # Split training data into train/val/test (70/10/20)
    train_df, temp_df = train_test_split(train_df, test_size=0.3, random_state=42, stratify=train_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.67, random_state=42, stratify=temp_df['label'])

    train_dataset = ClassificationDataset(
        train_df['image_path'].tolist(),
        train_df['label'].tolist(),
        transform=train_transform
    )

    val_dataset = ClassificationDataset(
        val_df['image_path'].tolist(),
        val_df['label'].tolist(),
        transform=val_transform
    )

    test_dataset = ClassificationDataset(
        test_df['image_path'].tolist(),
        test_df['label'].tolist(),
        transform=val_transform
    )

    logger.info(f"RSNA dataset loaded:")
    logger.info(f"  Training set: {len(train_dataset)} images")
    logger.info(f"  Validation set: {len(val_dataset)} images")
    logger.info(f"  Test set: {len(test_dataset)} images")

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'num_classes': num_classes
    }

