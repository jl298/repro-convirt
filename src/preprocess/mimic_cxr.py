import os
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from PIL import Image
import pydicom
from pathlib import Path


def extract_report_section(report, section_title):
    if not report:
        return ""

    pattern = f"{section_title}(.*?)(?:$|[A-Z]+:)"
    match = re.search(pattern, report, re.DOTALL)

    if match:
        return match.group(1).strip()
    return ""


def find_all_report_files(logger, reports_dir):
    report_dict = {}
    report_count = 0

    files_dir = os.path.join(reports_dir, 'files')
    if os.path.exists(files_dir):
        reports_dir = files_dir
        logger.info(f"Found 'files' subdirectory, using {reports_dir}")

    logger.info(f"Scanning for report files in {reports_dir}")
    for root, _, files in os.walk(reports_dir):
        for file in files:
            if file.endswith('.txt'):
                report_count += 1
                if file.startswith('s') and file.endswith('.txt'):
                    study_id = file.split('.')[0].lstrip('s')

                    try:
                        full_path = os.path.join(root, file)
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            report_content = f.read().strip()
                            if report_content:
                                report_dict[study_id] = report_content
                                if report_count % 1000 == 0:
                                    logger.info(
                                        f'Processed {report_count} reports, found {len(report_dict)} valid reports')
                    except UnicodeDecodeError:
                        logger.warning(f"Unicode error in report {file}")
                    except IOError:
                        logger.warning(f"IO error reading report {file}")

    logger.info(f"Found {report_count} total report files")
    logger.info(f"Successfully loaded {len(report_dict)} valid reports with study IDs")

    return report_dict


def process_split(logger, pairs, img_dir, report_dir):
    logger.info(f"Processing {len(pairs)} data pairs for {img_dir}")

    for pair in tqdm(pairs, desc="Saving data"):
        study_id = pair['study_id']

        report_path = os.path.join(report_dir, f"{study_id}.txt")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(pair['report'])
        except IOError as e:
            logger.warning(f"Failed to save report for study {study_id}: {e}")
            continue

        for j, img_path in enumerate(pair['image_paths']):
            if not os.path.exists(img_path):
                logger.warning(f"Image {img_path} not found")
                continue

            try:
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = Image.open(img_path)
                    dest_path = os.path.join(img_dir, f"{study_id}_{j}.jpg")
                    img.save(dest_path)

                elif img_path.lower().endswith('.dcm'):
                    dicom = pydicom.dcmread(img_path)
                    img_array = dicom.pixel_array

                    if img_array.max() > 0:
                        img_array = img_array / img_array.max() * 255
                    img_array = img_array.astype(np.uint8)

                    if len(img_array.shape) == 2:
                        img = Image.fromarray(img_array)
                    else:
                        img = Image.fromarray(img_array)

                    dest_path = os.path.join(img_dir, f"{study_id}_{j}.jpg")
                    img.save(dest_path)
            except IOError as e:
                logger.warning(f"Failed to process image {img_path}: {e}")
            except pydicom.errors.InvalidDicomError:
                logger.warning(f"Invalid DICOM file: {img_path}")
            except ValueError as e:
                logger.warning(f"Value error with {img_path}: {e}")


def create_metadata_json(logger, pairs, output_path):
    json_data = []

    for pair in pairs:
        study_id = pair['study_id']

        for j, _ in enumerate(pair['image_paths']):
            image_filename = f"{study_id}_{j}.jpg"

            json_data.append({
                "image_path": f"images/{image_filename}",
                "report": pair['report'],
                "study_id": study_id
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"Created metadata JSON at {output_path} with {len(json_data)} records")


def preprocess_mimic_cxr(logger, source_dir, reports_dir, output_dir, split_ratio, seed, max_samples=None):
    logger.info("Starting MIMIC-CXR dataset preparation")
    logger.info(f"Source data: {source_dir}")
    logger.info(f"Reports: {reports_dir}")

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'reports'), exist_ok=True)

    metadata_path = os.path.join(source_dir, 'mimic-cxr-2.0.0-metadata.csv')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata from {metadata_path}")
    else:
        csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
        if csv_files:
            metadata_path = os.path.join(source_dir, csv_files[0])
            metadata = pd.read_csv(metadata_path)
            logger.info(f"Found alternate metadata file: {metadata_path}")
        else:
            logger.error(f"No metadata file found in {source_dir}. Aborting.")
            return

    report_dict = find_all_report_files(logger, reports_dir)

    if len(report_dict) == 0:
        logger.error("No valid report files found. Cannot continue.")
        return

    processed_studies = set()
    train_pairs = []
    val_pairs = []
    test_pairs = []

    study_ids = metadata['study_id'].unique()
    if max_samples and max_samples < len(study_ids):
        study_ids = np.random.choice(study_ids, max_samples, replace=False)
        logger.info(f"Using a subset of {max_samples} samples for debugging")

    for study_id in tqdm(study_ids, desc="Processing studies"):
        study_id_str = str(study_id)

        if study_id_str in processed_studies:
            continue

        if study_id_str not in report_dict:
            continue

        study_rows = metadata[metadata['study_id'] == study_id]

        if len(study_rows) == 0:
            continue

        report_text = report_dict[study_id_str]

        findings_text = extract_report_section(report_text, "FINDINGS:")
        impression_text = extract_report_section(report_text, "IMPRESSION:")

        if not findings_text and not impression_text:
            continue

        report_content = ""
        if findings_text:
            report_content += "FINDINGS: " + findings_text + " "
        if impression_text:
            report_content += "IMPRESSION: " + impression_text

        if len(report_content.split()) < 10:
            continue

        split_random = np.random.random()
        if split_random < split_ratio['test']:
            target_split = 'test'
            target_pairs = test_pairs
        elif split_random < (split_ratio['test'] + split_ratio['val']):
            target_split = 'val'
            target_pairs = val_pairs
        else:
            target_split = 'train'
            target_pairs = train_pairs

        image_paths = []

        for _, row in study_rows.iterrows():
            try:
                dicom_id = str(row['dicom_id'])
                subject_id = str(row['subject_id'])

                jpg_path = os.path.join(source_dir, 'files', f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id_str}")
                if os.path.exists(jpg_path):
                    jpg_files = list(Path(jpg_path).glob(f"*{dicom_id}.jpg"))
                    if jpg_files:
                        image_paths.append(str(jpg_files[0]))
                        continue

                dcm_path = os.path.join(source_dir, 'files', f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id_str}",
                                        f"{dicom_id}.dcm")
                if os.path.exists(dcm_path):
                    image_paths.append(dcm_path)
            except (KeyError, IndexError) as e:
                logger.warning(f"Error finding image for study {study_id_str}, dicom {dicom_id}: {e}")

        if not image_paths:
            continue

        pair = {
            "study_id": study_id_str,
            "image_paths": image_paths,
            "report": report_content
        }

        target_pairs.append(pair)
        processed_studies.add(study_id_str)

        if len(processed_studies) % 100 == 0:
            logger.info(f"Processed {len(processed_studies)} studies")

    logger.info(f"Finished processing {len(processed_studies)} studies")
    logger.info(
        f"Created {len(train_pairs)} training pairs, {len(val_pairs)} validation pairs, and {len(test_pairs)} test pairs")

    process_split(logger, train_pairs, os.path.join(output_dir, 'train', 'images'), os.path.join(output_dir, 'train', 'reports'))
    process_split(logger, val_pairs, os.path.join(output_dir, 'val', 'images'), os.path.join(output_dir, 'val', 'reports'))
    process_split(logger, test_pairs, os.path.join(output_dir, 'test', 'images'), os.path.join(output_dir, 'test', 'reports'))

    create_metadata_json(logger, train_pairs, os.path.join(output_dir, 'chest_train_pairs.json'))
    create_metadata_json(logger, val_pairs, os.path.join(output_dir, 'chest_val_pairs.json'))
    create_metadata_json(logger, test_pairs, os.path.join(output_dir, 'chest_test_pairs.json'))

    logger.info("MIMIC-CXR dataset preparation complete")