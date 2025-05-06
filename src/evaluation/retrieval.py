import json
import os

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from common.dataset import load_retrieval_dataset
from preprocess.transforms import get_transforms_pretrain, get_transforms_classification
from utils.metrics import compute_similarity_matrix, compute_precision_at_k


def evaluate_retrieval(model, args, mode='both', logger=None):
    results = {}

    if mode in ['image2image', 'both']:
        logger.info("Evaluating image-to-image retrieval...")
        image2image_results = evaluate_image_to_image_retrieval(model, args, logger)
        results.update({f'image2image_{k}': v for k, v in image2image_results.items()})

    if mode in ['text2image', 'both']:
        logger.info("Evaluating text-to-image retrieval...")
        text2image_results = evaluate_text_to_image_retrieval(model, args, logger)
        results.update({f'text2image_{k}': v for k, v in text2image_results.items()})

    results_path = os.path.join(args.output_dir, 'retrieval_results.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f)
    logger.info(f"Retrieval results saved to {results_path}")

    return results

def evaluate_image_to_image_retrieval(model, args, logger):
    data = load_retrieval_dataset(args.eval_data_dir, args.img_base_dir, 'retrieval_img')
    retrieval_dataset = data['dataset']

    logger.info("Processing query images...")
    query_data = retrieval_dataset.query_data
    _, val_transform = get_transforms_classification()

    query_images = []
    query_labels = []

    for item in tqdm(query_data, desc="Preparing query images"):
        image = retrieval_dataset.get_query(query_data.index(item))
        if isinstance(image, torch.Tensor):
            query_images.append(image)
        else:
            image = val_transform(image)
            query_images.append(image)
        query_labels.append(item['label'])

    query_images = torch.stack(query_images)
    query_dataloader = [(query_images, query_labels)]

    return perform_retrieval_evaluation(
        model=model,
        args=args,
        query_dataloader=query_dataloader,
        query_labels=query_labels,
        candidate_data=retrieval_dataset.candidate_data,
        retrieval_dataset=retrieval_dataset,
        mode='image2image',
        logger=logger
    )

def evaluate_text_to_image_retrieval(model, args, logger):
    data = load_retrieval_dataset(args.eval_data_dir, args.img_base_dir, 'retrieval_txt')
    retrieval_dataset = data['dataset']

    logger.info("Processing query text...")
    query_data = retrieval_dataset.query_data

    query_inputs = []
    query_masks = []
    query_labels = []

    for item in tqdm(query_data, desc="Preparing query text"):
        input_ids, attention_mask = retrieval_dataset.get_query(query_data.index(item))
        query_inputs.append(input_ids)
        query_masks.append(attention_mask)
        query_labels.append(item['label'])

    query_inputs = torch.stack(query_inputs)
    query_masks = torch.stack(query_masks)
    query_dataloader = [((query_inputs, query_masks), query_labels)]

    return perform_retrieval_evaluation(
        model=model,
        args=args,
        query_dataloader=query_dataloader,
        query_labels=query_labels,
        candidate_data=retrieval_dataset.candidate_data,
        retrieval_dataset=retrieval_dataset,
        mode='text2image',
        logger=logger
    )

def perform_retrieval_evaluation(model, args, query_dataloader, query_labels,
                                 candidate_data, retrieval_dataset, mode, logger):
    top_k_values = getattr(args, 'top_k', [1, 5, 10, 50])

    logger.info(f"Extracting {'image' if mode == 'image2image' else 'text'} embeddings...")
    query_embeddings = extract_embeddings(model, query_dataloader, args.device, is_text=(mode == 'text2image'))
    if isinstance(query_embeddings, tuple):
        query_embeddings, _ = query_embeddings

    candidate_embeddings, candidate_labels = process_candidates_in_batches(
        model=model,
        candidate_data=candidate_data,
        transform=retrieval_dataset.transform,
        device=args.device,
        batch_size=64,
        is_text=(mode == 'text2image'),
        logger=logger
    )

    if candidate_embeddings is None:
        logger.error("Failed to extract candidate embeddings!")
        return {}

    similarity_matrix = compute_similarity_matrix(query_embeddings, candidate_embeddings)

    results = {}
    for k in top_k_values:
        if k <= len(candidate_labels):
            p_at_k = compute_precision_at_k(similarity_matrix, query_labels, candidate_labels, k)
            results[f'p@{k}'] = float(p_at_k)
            logger.info(f"{mode} P@{k}: {p_at_k:.4f}")
        else:
            logger.warning(f"k={k} is larger than number of candidates ({len(candidate_labels)})")

    if getattr(args, 'save_individual', False):
        results_path = os.path.join(args.output_dir, f'results_retrieval_{mode}.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({"task": f"retrieval_{mode}", **results}, f)
        logger.info(f"Results saved to {results_path}")

    return results

def extract_embeddings(model, dataloader, device, is_text=False):
    model.to(device)
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {'text' if is_text else 'image'} embeddings"):
            if is_text:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    batch_labels = batch.get('labels')

                    text_features = model.encode_text(input_ids, attention_mask)
                    proj_features = model.txt_projection(text_features)
                    proj_features = torch.nn.functional.normalize(proj_features, p=2, dim=1)
                else:
                    text_data, batch_labels = batch[0], batch[1]
                    input_ids = text_data[0].to(device)
                    attention_mask = text_data[1].to(device)

                    text_features = model.encode_text(input_ids, attention_mask)
                    proj_features = model.txt_projection(text_features)
                    proj_features = torch.nn.functional.normalize(proj_features, p=2, dim=1)
            else:
                if isinstance(batch, dict):
                    images = batch['images'].to(device)
                    batch_labels = batch.get('labels')
                else:
                    images = batch[0].to(device)
                    batch_labels = batch[1] if len(batch) > 1 else None

                image_features = model.encode_image(images)
                # Removing the project head, do not use model.vis_projection
                proj_features = torch.nn.functional.normalize(image_features, p=2, dim=1)

            embeddings.append(proj_features.cpu().numpy())

            if batch_labels is not None:
                if isinstance(batch_labels, torch.Tensor):
                    labels.append(batch_labels.cpu().numpy())
                else:
                    labels.append(batch_labels)

    embeddings = np.concatenate(embeddings, axis=0)

    if labels:
        if isinstance(labels[0], np.ndarray):
            labels = np.concatenate(labels, axis=0)
        return embeddings, labels

    return embeddings

def process_candidates_in_batches(model, candidate_data, transform, device, batch_size=16, is_text=False, logger=None):
    candidate_embeddings = []
    candidate_labels = []

    if logger:
        logger.info(f"Processing {len(candidate_data)} candidate images in batches of {batch_size}...")
    else:
        print(f"Processing {len(candidate_data)} candidate images in batches of {batch_size}...")

    for i in tqdm(range(0, len(candidate_data), batch_size), desc="Processing candidate batches"):
        batch_data = candidate_data[i:i + batch_size]

        batch_images = []
        batch_labels = []

        for item in batch_data:
            try:
                image_path = item['image_path']
                image = Image.open(image_path).convert('RGB')

                if transform:
                    image = transform(image)

                batch_images.append(image)
                batch_labels.append(item['label'])
            except Exception as e:
                if logger:
                    logger.warning(f"Error loading candidate image {image_path}: {e}")
                else:
                    print(f"Error loading candidate image {image_path}: {e}")
                continue

        if not batch_images:
            continue

        batch_images = torch.stack(batch_images).to(device)

        with torch.no_grad():
            image_features = model.encode_image(batch_images)

            if is_text:
                # image-to-image removes projection head
                # text-to-image keeps projection head
                proj_features = model.vis_projection(image_features)
            proj_features = torch.nn.functional.normalize(proj_features, p=2, dim=1)

            candidate_embeddings.append(proj_features.cpu().numpy())
            candidate_labels.extend(batch_labels)

    if candidate_embeddings:
        candidate_embeddings = np.concatenate(candidate_embeddings, axis=0)
        if logger:
            logger.info(f"Extracted embeddings for {len(candidate_labels)} candidates.")
        else:
            print(f"Extracted embeddings for {len(candidate_labels)} candidates.")
        return candidate_embeddings, candidate_labels
    else:
        if logger:
            logger.error("No valid candidate embeddings were extracted.")
        else:
            print("No valid candidate embeddings were extracted.")
        return None, []
