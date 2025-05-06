import os
import argparse
import torch
from torch.cuda.amp import autocast

from common.model import ConVIRTModel, ClassificationModel, BaselineModel
from evaluation.downstream_amp import evaluate_classification, get_class_names
from visualization.classification_charts_fixed import ClassificationVisualizer
from utils.logger import get_logger_downstream
from utils import set_seed
from utils.metrics import classification_result


def parse_args():
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--task', required=True, type=str, choices=['rsna', 'chexpert', 'covidx', 'mura'])
    parser.add_argument('--mode', default='linear', type=str, choices=['linear', 'finetune'])
    parser.add_argument('--percent', default=100, type=int, choices=[1, 10, 100])
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--lambda_weight', default=0.75, type=float)
    parser.add_argument('--proj_dim', default=512, type=int)
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--saliency', action='store_true')
    parser.add_argument('--num_saliency_samples', default=4, type=int)

    args = parser.parse_args()
    return args


def collect_test_samples(test_loader, num_samples, device, use_amp=False):
    collected_images = []
    collected_labels = []
    total_collected = 0

    with torch.no_grad():
        for images, labels in test_loader:
            batch_size = images.size(0)

            if total_collected + batch_size <= num_samples:
                if use_amp:
                    with autocast():
                        collected_images.append(images.to(device))
                else:
                    collected_images.append(images.to(device))
                collected_labels.append(labels.to(device))
                total_collected += batch_size
            else:
                remaining = num_samples - total_collected
                if use_amp:
                    with autocast():
                        collected_images.append(images[:remaining].to(device))
                else:
                    collected_images.append(images[:remaining].to(device))
                collected_labels.append(labels[:remaining].to(device))
                break

    return torch.cat(collected_images, dim=0), torch.cat(collected_labels, dim=0)


def create_comparison_models(args, best_classifier, dataset, logger):
    param = next(best_classifier.parameters())
    if param.dtype == torch.float16:
        best_classifier = best_classifier.float()

    models_dict = {'ConVIRT': best_classifier}

    if args.saliency:
        baseline = BaselineModel(init_type='imagenet')
        model = ClassificationModel(
            encoder=baseline.get_image_encoder(),
            num_classes=dataset['num_classes'],
            freeze_encoder=(args.mode == 'linear')
        ).to(args.device)

        models_dict['ImageNet'] = model
        logger.info("Created ImageNet comparison model using BaselineModel")

        baseline = BaselineModel(init_type='random')
        model = ClassificationModel(
            encoder=baseline.get_image_encoder(),
            num_classes=dataset['num_classes'],
            freeze_encoder=(args.mode == 'linear')
        ).to(args.device)

        models_dict['Random'] = model
        logger.info("Created Random initialization comparison model using BaselineModel")

    return models_dict


def create_visualizations(args, outputs_dict, best_classifier, dataset, logger):
    if not any([args.tsne, args.saliency]):
        return

    logger.info("Creating visualizations...")

    task_output_dir = args.output_dir
    if os.path.basename(args.output_dir) != args.task:
        task_output_dir = os.path.join(args.output_dir, args.task)

    viz_dir = os.path.join(task_output_dir, args.mode, str(args.percent), 'visualizations')
    logger.info(f"Creating visualization directory at: {viz_dir}")
    os.makedirs(viz_dir, exist_ok=True)

    class_names = get_class_names(args.task, dataset['num_classes'])
    logger.info(f"Class names for {args.task}: {class_names}")

    class_visualizer = ClassificationVisualizer(viz_dir)

    if args.tsne and 'embeddings' in outputs_dict:
        logger.info("Creating t-SNE visualization...")
        class_visualizer.plot_tsne(
            embeddings=outputs_dict['embeddings'],
            labels=outputs_dict['labels'],
            class_names=class_names,
            title=f't-SNE - {args.task.upper()} ({args.mode}, {args.percent}%)',
            save_path=os.path.join(viz_dir, f'tsne_{args.task}_{args.mode}_{args.percent}.png')
        )

    if 'y_true' in outputs_dict and 'y_pred' in outputs_dict:
        logger.info("Creating classification visualizations...")

        class_visualizer.plot_confusion_matrix(
            y_true=outputs_dict['y_true'],
            y_pred=outputs_dict['y_pred'],
            class_names=class_names,
            title=f'Confusion Matrix - {args.task.upper()}',
            save_path=os.path.join(viz_dir, f'confusion_matrix_{args.task}.png')
        )

        if 'y_score' in outputs_dict:
            class_visualizer.plot_roc_curves(
                y_true=outputs_dict['y_true'],
                y_score=outputs_dict['y_score'],
                class_names=class_names,
                title=f'ROC Curves - {args.task.upper()}',
                save_path=os.path.join(viz_dir, f'roc_curves_{args.task}.png')
            )

    if args.saliency:
        test_loader = dataset['test_loader']
        sample_images, sample_labels = collect_test_samples(
            test_loader, args.num_saliency_samples, args.device, args.use_amp
        )

        logger.info("Creating basic saliency maps...")
        if args.use_amp:
            with autocast():
                class_visualizer.plot_saliency_maps(
                    model=best_classifier,
                    images=sample_images,
                    labels=sample_labels,
                    class_names=class_names,
                    device=args.device,
                    save_path=os.path.join(viz_dir, 'basic_saliency_maps.png')
                )
        else:
            class_visualizer.plot_saliency_maps(
                model=best_classifier,
                images=sample_images,
                labels=sample_labels,
                class_names=class_names,
                device=args.device,
                save_path=os.path.join(viz_dir, 'basic_saliency_maps.png')
            )

        logger.info("Creating advanced saliency maps...")
        baseline_models_dict = create_comparison_models(args, best_classifier, dataset, logger)

        if args.task == 'chexpert':
            class_visualizer.visualize_chexpert_abnormalities(
                models_dict=baseline_models_dict,
                chexpert_loader=test_loader,
                abnormality_classes=class_names,
                num_samples_per_class=1
            )
        else:
            class_visualizer.create_paper_figure(
                models_dict=baseline_models_dict,
                dataset_loader=test_loader,
                class_names=class_names,
                num_samples=args.num_saliency_samples
            )

        saliency_dir = os.path.join(viz_dir, 'saliency')
        os.makedirs(saliency_dir, exist_ok=True)

        for i in range(len(sample_images)):
            with torch.no_grad():
                image = sample_images[i].unsqueeze(0)
                # Handle multi-label tensors properly - no .item() call
                label = sample_labels[i]

            class_visualizer.visualize_comparison(
                models_dict=baseline_models_dict,
                image_tensor=image,
                true_label=label,
                class_names=class_names,
                save_path=os.path.join(saliency_dir, f'comparison_{i}.png')
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(f"Visualizations saved to {viz_dir}")


def main():
    set_seed()
    args = parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.use_amp and not torch.cuda.is_available():
        args.use_amp = False
        print("AMP is only available on CUDA devices. Disabling AMP.")

    if os.path.basename(args.output_dir) == args.task:
        task_output_dir = args.output_dir
    else:
        task_output_dir = os.path.join(args.output_dir, args.task)

    os.makedirs(task_output_dir, exist_ok=True)

    if args.log_dir is None:
        args.log_dir = os.path.join(task_output_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Log directory set to: {args.log_dir}")
    if args.use_amp:
        print("Using Automatic Mixed Precision (AMP)")

    logger = get_logger_downstream(args.log_dir, args.task, args.mode, args.percent)
    logger.info(f"Arguments: {args}")

    logger.info("Creating model...")
    model = ConVIRTModel(
        img_encoder_name="resnet50",
        txt_encoder_name="emilyalsentzer/Bio_ClinicalBERT",
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        lambda_weight=args.lambda_weight
    ).to(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f'Loaded checkpoint from {args.checkpoint}')

    args.eval_task = args.task
    args.eval_percent = args.percent
    args.eval_mode = args.mode

    if args.use_amp:
        with autocast():
            metrics, outputs_dict, best_classifier, dataset = evaluate_classification(model, args, logger)
    else:
        metrics, outputs_dict, best_classifier, dataset = evaluate_classification(model, args, logger)

    print(f"Results: {metrics}")
    classification_result(args, metrics, outputs_dict)
    create_visualizations(args, outputs_dict, best_classifier, dataset, logger)

    print(f"Evaluation finished for {args.task} ({args.mode}, {args.percent}%)")


if __name__ == '__main__':
    main()
