# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import transformers
import torchvision.transforms as transforms
import argparse
import tqdm
import random
import json
from open_clip import create_model_and_transforms, get_tokenizer
from optimization_utils import (
    Scorer as S,
    Generator as G,
    get_text_features,
    get_image_features,
)

from paths import IMAGEC_COCO_ANNOTATIONS, IMAGEC_COCO_IMAGES, IMAGEC_COCO_SPLITS, OUTPUT_DIR


def display_single_image_with_caption(image_id, images_path, best_caption, score):
    """Display a single image with its best caption in real-time"""
    # Load and display image
    if not images_path.endswith('val2014'):
        image_path = os.path.join(images_path, 'val2014', f"COCO_val2014_{image_id:012}.jpg")
    else:
        image_path = os.path.join(images_path, f"COCO_val2014_{image_id:012}.jpg")
    
    try:
        # Use proper Colab display method
        show_image_colab(image_path, 
                        f"Image ID: {image_id}\nScore: {score:.3f}\n{best_caption}", 
                        figsize=(8, 6))
        
        print(f"Image {image_id} - Current best caption:")
        print(f"Score: {score:.3f}")
        print(f"Caption: {best_caption}")
        print("-" * 60)
        
    except Exception as e:
        print(f"Error displaying image {image_id}: {e}")


def show_image_colab(image_path, title="Image", figsize=(8, 6)):
    """Properly display images in Colab"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        from IPython.display import display
        
        # Configure matplotlib for Colab
        if 'google.colab' in str(get_ipython()):
            matplotlib.use('Agg')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Load and display image
        from PIL import Image
        img = Image.open(image_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=12, pad=10)
        
        # Show plot
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"Could not display image: {e}")
        print(f"Image path: {image_path}")
        return False


def display_initial_images_grid(image_ids, images_path, max_images=16):
    """Display all images in a grid before optimization starts"""
    # Configure matplotlib for Colab display
    current_backend = matplotlib.get_backend()
    matplotlib.use('module://ipykernel.pylab.backend_inline')
    plt.ion()
    
    try:
        n_images = min(len(image_ids), max_images)
        print(f"Displaying {n_images} images to be optimized...")
        
        # Determine grid size - aim for roughly square grid
        if n_images <= 4:
            rows, cols = 2, 2
        elif n_images <= 6:
            rows, cols = 2, 3
        elif n_images <= 9:
            rows, cols = 3, 3
        elif n_images <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if n_images == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, image_id in enumerate(image_ids[:max_images]):
            # Load image
            if not images_path.endswith('val2014'):
                image_path = os.path.join(images_path, 'val2014', f"COCO_val2014_{image_id:012}.jpg")
            else:
                image_path = os.path.join(images_path, f"COCO_val2014_{image_id:012}.jpg")
            
            try:
                img = Image.open(image_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f"Image {image_id}", fontsize=10, pad=5)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\nImage {image_id}", 
                            ha='center', va='center', transform=axes[i].transAxes, fontsize=8)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Images to be optimized ({n_images} total)", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()
        
        print(f"✅ Displayed {n_images} images in grid format")
        
    except Exception as e:
        print(f"Error displaying images grid: {e}")
    finally:
        # Reset to previous backend
        matplotlib.use(current_backend)
        plt.ioff()


def display_images_with_captions(image_ids, images_path, output_dir, max_images=8):
    """Display images with their final optimized captions"""
    try:
        # Configure matplotlib for proper display
        current_backend = matplotlib.get_backend()
        matplotlib.use('module://ipykernel.pylab.backend_inline')
        plt.ion()
        
        n_images = min(len(image_ids), max_images)
        if n_images == 0:
            print("No images to display")
            return
            
        print(f"Creating image comparison for {n_images} images...")
        
        # Create figure with proper sizing
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        
        # Handle single subplot case
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
        
        # Process each image
        for i, image_id in enumerate(image_ids[:max_images]):
            try:
                # Load image
                if not images_path.endswith('val2014'):
                    image_path = os.path.join(images_path, 'val2014', f"COCO_val2014_{image_id:012}.jpg")
                else:
                    image_path = os.path.join(images_path, f"COCO_val2014_{image_id:012}.jpg")
                
                if os.path.exists(image_path):
                    img = Image.open(image_path)
                    axes[i].imshow(img)
                else:
                    # Create placeholder if image missing
                    placeholder = np.ones((224, 224, 3)) * 0.8
                    axes[i].imshow(placeholder)
                    axes[i].text(0.5, 0.5, f'Image {image_id}\nNot Found', 
                               ha='center', va='center', transform=axes[i].transAxes)
                
                axes[i].axis('off')
                
                # Get final caption from log file
                log_path = os.path.join(output_dir, f"{image_id}", "log.txt")
                final_caption = "No caption available"
                final_score = 0.0
                
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                # Get last line (final result)
                                last_line = lines[-1].strip()
                                parts = last_line.split('\t')
                                if len(parts) >= 3:
                                    final_score = float(parts[0])
                                    final_caption = parts[2]
                    except Exception as e:
                        print(f"Error reading log for image {image_id}: {e}")
                
                # Set title with caption
                title = f"Image {image_id}\nScore: {final_score:.3f}\n{final_caption[:50]}..."
                axes[i].set_title(title, fontsize=8, pad=5)
                
            except Exception as e:
                print(f"Error processing image {image_id}: {e}")
                axes[i].text(0.5, 0.5, f'Error\nImage {image_id}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Final Image Captions ({n_images} images)", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save comparison image
        comparison_path = os.path.join(output_dir, "image_caption_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"Image comparison saved to: {comparison_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating image comparison: {e}")
    finally:
        # Reset matplotlib backend
        matplotlib.use(current_backend)
        plt.ioff()


def show_optimization_progress(image_ids, output_dir, max_images=4):
    """Show optimization progress with better error handling"""
    try:
        # Configure matplotlib for Colab display
        current_backend = matplotlib.get_backend()
        matplotlib.use('module://ipykernel.pylab.backend_inline')
        plt.ion()
        
        print(f"Creating optimization progress charts for {len(image_ids)} images...")
        
        # Determine grid size
        n_images = min(len(image_ids), max_images)
        cols = min(2, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
        if n_images == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
        
        has_data = False
        
        for i, image_id in enumerate(image_ids[:max_images]):
            log_path = os.path.join(output_dir, f"{image_id}", "log.txt")
            
            if os.path.exists(log_path):
                try:
                    scores = []
                    avg_scores = []
                    
                    with open(log_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                try:
                                    scores.append(float(parts[0]))
                                    avg_scores.append(float(parts[1]))
                                except ValueError:
                                    continue
                    
                    if scores:
                        has_data = True
                        iterations = range(len(scores))
                        
                        # Plot scores
                        axes[i].plot(iterations, scores, 'b-o', label='Best Score', linewidth=2, markersize=4)
                        axes[i].plot(iterations, avg_scores, 'r--s', label='Avg Score', linewidth=2, markersize=4)
                        
                        # Calculate improvement
                        if len(scores) > 1:
                            improvement = scores[-1] - scores[0]
                            axes[i].set_title(f'Image {image_id}\nImprovement: +{improvement:.4f}', fontsize=10)
                        else:
                            axes[i].set_title(f'Image {image_id}\nSingle Iteration', fontsize=10)
                        
                        axes[i].set_xlabel('Iteration')
                        axes[i].set_ylabel('CLIP Score')
                        axes[i].legend(fontsize=8)
                        axes[i].grid(True, alpha=0.3)
                        
                        # Set y-axis to show small improvements
                        y_min, y_max = min(min(scores), min(avg_scores)), max(max(scores), max(avg_scores))
                        y_range = y_max - y_min
                        if y_range < 0.01:  # Very small range
                            y_center = (y_min + y_max) / 2
                            axes[i].set_ylim(y_center - 0.01, y_center + 0.01)
                        
                    else:
                        axes[i].text(0.5, 0.5, f'No valid data\nfor Image {image_id}', 
                                   ha='center', va='center', transform=axes[i].transAxes, fontsize=10)
                        
                except Exception as e:
                    print(f"Error processing log for image {image_id}: {e}")
                    axes[i].text(0.5, 0.5, f'Error loading data\nfor Image {image_id}', 
                               ha='center', va='center', transform=axes[i].transAxes, fontsize=10)
            else:
                axes[i].text(0.5, 0.5, f'No log file found\nfor Image {image_id}', 
                           ha='center', va='center', transform=axes[i].transAxes, fontsize=10)
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        if has_data:
            plt.suptitle(f'Optimization Progress ({n_images} images)', fontsize=14, y=0.98)
        else:
            plt.suptitle(f'No Progress Data Available ({n_images} images)', fontsize=14, y=0.98)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save progress chart
        progress_path = os.path.join(output_dir, "optimization_progress.png")
        plt.savefig(progress_path, dpi=150, bbox_inches='tight')
        print(f"Optimization progress saved to: {progress_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating optimization progress chart: {e}")
    finally:
        # Reset to previous backend
        matplotlib.use(current_backend)
        plt.ioff()


def optimize_for_images(
    args, text_pipeline, image_ids, text_prompt, clip_model, tokenizer, preprocess
):
    loggers = {}
    save_locations = {}
    for image_id in image_ids:
        save_locations[f"{image_id}"] = os.path.join(args.output_dir, f"{image_id}")
        os.makedirs(save_locations[f"{image_id}"], exist_ok=True)
        loggers[f"{image_id}"] = open(
            os.path.join(save_locations[f"{image_id}"], "log.txt"), "w"
        )

    # debugging
    test_outputs = text_pipeline(
        text_prompt,
        max_new_tokens=50,
        num_return_sequences=args.requested_number,
    )
    print(f"\n--- RAW GENERATIONS for {args.text_model} ---")
    for o in test_outputs:
        print(o["generated_text"])
    print("-------------------------\n")


    generator = G(
        text_pipeline,
        args.text_model,
        requested_number=args.requested_number,
        keep_previous=args.keep_previous,
        prompt=text_prompt,
        key=lambda x: -x[0],
        batch_size=args.batch_size,
        device=args.device,
        exploration=args.exploration,
        verbose=1,  # Enable verbose mode to see debug info
    )
    image_paths = [
        os.path.join(args.images_path, "val2014", f"COCO_val2014_{image_id:012}.jpg")
        for image_id in image_ids
    ]
    target_features = (
        get_image_features(
            clip_model,
            preprocess,
            image_paths,
            args.device,
            args.batch_size,
        )
        .detach()
        .cpu()
        .numpy()
    )

    def clip_scorer(sentences, target_feature):
        text_features = get_text_features(
            clip_model,
            tokenizer,
            sentences,
            args.device,
            args.batch_size,
            amp=True,
            use_format=False,
        )
        return text_features.detach().cpu().numpy() @ target_feature

    scorers = {}
    for i, image_id in enumerate(image_ids):
        scorers[f"{image_id}"] = {
            "func": clip_scorer,
            "target_feature": target_features[i],
        }

    scorer = S(
        scorers, args.batch_size, key=lambda x: -x, keep_previous=args.keep_previous
    )
    ###
    # Initialize the pool
    ###
    with open(args.init_descriptions, "r") as w:
        init_sentences = [i.strip() for i in w.readlines()]
    if args.init_descriptions_set_size != "all":
        random.seed(0) # Choose a different seed than args as it is already used (should not matter though)
        init_sentences = random.sample(init_sentences, int(args.init_descriptions_set_size)) # Must be all or an int
    lines_with_scores = {}
    initial_scores = {}
    
    # Display all images together first, before optimization
    print("=" * 60)
    print("DISPLAYING ALL IMAGES TO BE OPTIMIZED:")
    print("=" * 60)
    display_initial_images_grid(image_ids, args.images_path)
    
    for i, image_id in enumerate(image_ids):
        print(f"Scoring initial sentences for image {image_id}...")
        init_scores = scorer.score(f"{image_id}", init_sentences)
        lines_with_scores[f"{image_id}"] = [
            (s, l) for (s, l) in zip(init_scores, init_sentences)
        ]
        best_score = sorted(lines_with_scores[f"{image_id}"], key=lambda x: -x[0])[0]
        initial_scores[f"{image_id}"] = best_score
        mean_score = np.mean(init_scores)
        bs = best_score[1].strip()
        loggers[f"{image_id}"].write(f"{best_score[0]}\t{mean_score}\t{bs}\n")
        
        # Print initial best caption (without displaying individual images)
        print(f"Image {image_id} initial best: {best_score[0]:.4f} - {bs}")
        
    ###
    # Do the optimization:
    ###
    print("\n" + "=" * 60)
    print("STARTING OPTIMIZATION PROCESS:")
    print("=" * 60)
    
    previous_best_scores = {key: initial_scores[key][0] for key in initial_scores}
    
    for it in range(args.iterations):
        print(f"\n{'='*20} ITERATION {it+1}/{args.iterations} {'='*20}")
        torch.cuda.empty_cache()
        
        # Check if GPT-2 generation is working properly
        print("Generating new descriptions...")
        new_lines = generator(lines_with_scores)
        
        # Check if we got meaningful results
        total_new_descriptions = sum(len(descriptions) for descriptions in new_lines.values())
        print(f"Generated {total_new_descriptions} total new descriptions")
        
        if total_new_descriptions == 0:
            print("WARNING: No new descriptions generated! Stopping optimization.")
            break
        
        print(f"Scoring new descriptions...")
        lines_with_scores = scorer(new_lines)
        
        best_value = scorer.get_best_value()  # Text to score
        best = scorer.get_best()  # Text to (text, image)
        average_value = scorer.get_average_value()  # Text to score
        
        # Display progress for each image
        for key in average_value:
            current_score = best_value[key][0]
            improvement = current_score - previous_best_scores[key]
            
            loggers[key].write(
                f"{current_score}\t{average_value[key]}\t{best[key]}\n"
            )
            
            print(f"Image {key}: {current_score:.4f} (+{improvement:.4f}) - {best[key]}")
            
            previous_best_scores[key] = current_score
        
        # Early stopping if no improvement
        if it > 0:  # Skip first iteration
            total_improvement = sum(best_value[key][0] - initial_scores[key][0] for key in best_value)
            print(f"\nTotal improvement since start: {total_improvement:.4f}")
            
            if total_improvement < 0.001:  # Very small improvement threshold
                print("Minimal improvement detected. Stopping early.")
                break
    
    for k, logger in loggers.items():
        logger.close()
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)


def main(args):
    # Set seeds
    torch.manual_seed(args.seed)

    # Load text prompt and annotations
    with open(args.prompt, "r") as f:
        text_prompt = f.read()
    with open(args.annotations_path, "r") as f:
        annotations = json.load(f)["annotations"]

    # Load CLIP model
    clip_model, _, preprocess = create_model_and_transforms(args.clip_model, pretrained=args.pretrained)
    tokenizer = get_tokenizer(args.clip_model)
    clip_model.to(args.device)
    clip_model.eval()

    # Load text generation model
    print(f" 🤖 Loading text model: {args.text_model}")
    text_pipeline = transformers.pipeline(
        "text-generation",
        model=args.text_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=args.device,
        trust_remote_code=True,
    )
    print(f" ✅ Model ready: {args.text_model}")

    # Ensure tokenizer has pad_token
    if text_pipeline.tokenizer.pad_token is None:
        text_pipeline.tokenizer.pad_token = text_pipeline.tokenizer.eos_token or "<pad>"
    if getattr(text_pipeline.model.config, 'pad_token_id', None) is None:
        text_pipeline.model.config.pad_token_id = text_pipeline.tokenizer.pad_token_id

    # Fixed set of 16 COCO image IDs
    fixed_image_ids = [
        442539, 170898, 544857, 285820, 188414, 385248, 249227, 502311,
        391895, 397133, 37777, 252219, 87038, 174482, 403385, 6818
    ]
    
    # Only keep the ones that actually exist in the folder
    images_dir = args.images_path
    if not images_dir.endswith("val2014"):
        images_dir = os.path.join(images_dir, "val2014")
    
    image_ids = [
        img_id for img_id in fixed_image_ids
        if os.path.exists(os.path.join(images_dir, f"COCO_val2014_{img_id:012}.jpg"))
    ]
    
    # Skip ones already processed
    image_ids = [
        x for x in image_ids
        if not os.path.exists(os.path.join(args.output_dir, f"{x}"))
    ]
    
    # Respect process splitting (if running multi-process)
    image_ids = image_ids[args.process :: args.num_processes]


    # Process images in batches
    while len(image_ids):
        current_batch = []
        while len(current_batch) < args.llm_batch_size and image_ids:
            image_id = image_ids.pop(0)
            if not os.path.exists(os.path.join(args.output_dir, f"{image_id}")):
                current_batch.append(image_id)

        if current_batch:
            optimize_for_images(
                args,
                text_pipeline,
                current_batch,
                text_prompt,
                clip_model,
                tokenizer,
                preprocess,
            )


def get_args_parser():
    parser = argparse.ArgumentParser("Image Captioning with COCO", add_help=False)

    # Model parameters
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--device", default="cuda:0", help="device to use for testing")
    parser.add_argument(
        "--annotations_path",
        default=IMAGEC_COCO_ANNOTATIONS,
    )
    parser.add_argument(
        "--images_path", default=IMAGEC_COCO_IMAGES
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR,
        help="Output Path",
    )
    parser.add_argument("--num_processes", default=1, type=int)
    parser.add_argument("--process", default=0, type=int)
    

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--llm_batch_size", default=16, type=int, help="Batch size for llms"
    )
    parser.add_argument("--keep_previous", default=25, type=int, help="Keep previous")
    parser.add_argument(
        "--requested_number", default=20, type=int, help="How many to request"
    )
    parser.add_argument(
        "--iterations", default=10, type=int, help="Optimization iterations"
    )
    parser.add_argument(
        "--clip_model",
        default="ViT-B-32", #SMALLER VISUAL MODEL
        type=str,
        metavar="MODEL",
        help="Name of model to use",
    )
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k", type=str)

    parser.add_argument(
        "--text_model",
        default="gpt2",
        type=str,
        help="Hugging Face text model for caption generation. Examples:\n"
             "  - gpt2 (default, fast but basic)\n"
             "  - microsoft/DialoGPT-medium (better quality)\n"
             "  - Qwen/Qwen2.5-0.5B (instruction-tuned)\n"
             "  - MBZUAI/MobiLlama-05B (efficient mobile model)\n"
             "  - microsoft/phi-2 (small but capable)\n"
             "  - Any other Hugging Face text generation model"
    )
    parser.add_argument(
        "--init_descriptions",
        default="init_descriptions/image_descriptions_per_class.txt",
        type=str,
        help="init descriptions pool",
    )
    parser.add_argument(
        "--init_descriptions_set_size",
        default="all",
        type=str,
        help="How many descriptions to choose, should be either int or an int",
    )
    parser.add_argument(
        "--prompt", default="prompts/image_captioning_shorter.txt", type=str, help="Prompt"
    )
    parser.add_argument("--exploration", default=0.0, type=float, help="exploration")
    # Dataset parameters
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no_ablation", action="store_false", dest="ablation")
    parser.set_defaults(ablation=False)
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    text_model = args.text_model.split("/")[-1].replace("-", "-")
    prompt = args.prompt.split("/")[-1].split(".")[0]
    name = "imagec_g" if not args.ablation else "imagec_a"
    args.output_dir = os.path.join(
        args.output_dir,
        f"{name}_{text_model}_{args.iterations}_{args.exploration}_{args.keep_previous}_{args.requested_number}_{args.clip_model}_{args.pretrained}_{prompt}_{args.init_descriptions_set_size}",
    )
    print(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
