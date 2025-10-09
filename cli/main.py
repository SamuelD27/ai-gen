"""
CharForgeX unified CLI interface.
Complete control over LoRA training and media generation.
"""

import click
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_config, get_config


@click.group()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.pass_context
def cli(ctx, config):
    """
    CharForgeX - Professional LoRA Training & Media Generation

    No content restrictions. Maximum creative freedom.
    """
    # Load configuration
    load_config(config)
    ctx.ensure_object(dict)
    ctx.obj['config'] = get_config()


# Dataset commands
@cli.group()
def dataset():
    """Dataset processing commands."""
    pass


@dataset.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--min-resolution', default=512, help='Minimum image resolution')
@click.option('--require-faces/--no-require-faces', default=True, help='Only keep images with faces')
@click.option('--quality-filter/--no-quality-filter', default=True, help='Enable quality filtering')
def clean(input_dir, output_dir, min_resolution, require_faces, quality_filter):
    """Clean dataset: remove low quality, duplicates, etc."""
    from dataset.cleaning import quick_clean

    click.echo(f"Cleaning dataset from {input_dir} to {output_dir}")

    stats = quick_clean(
        input_dir=input_dir,
        output_dir=output_dir,
        min_resolution=min_resolution,
        require_faces=require_faces,
        quality_filter=quality_filter,
    )

    click.echo(f"\nCleaning complete:")
    click.echo(f"  Valid images: {stats['valid']}/{stats['total']}")


@dataset.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--resolution', default=1024, help='Target resolution')
@click.option('--smart-crop/--center-crop', default=True, help='Use face-aware cropping')
@click.option('--augment/--no-augment', default=False, help='Create augmented versions')
def preprocess(input_dir, output_dir, resolution, smart_crop, augment):
    """Preprocess dataset: resize, crop, augment."""
    from dataset.preprocessing import quick_preprocess

    click.echo(f"Preprocessing dataset from {input_dir} to {output_dir}")

    stats = quick_preprocess(
        input_dir=input_dir,
        output_dir=output_dir,
        resolution=resolution,
        smart_crop=smart_crop,
        augment=augment,
    )

    click.echo(f"\nPreprocessing complete:")
    click.echo(f"  Output images: {stats['total_output']}")


@dataset.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--backend', type=click.Choice(['blip2', 'clip', 'cogvlm', 'ensemble']), default='blip2')
@click.option('--batch-size', default=4, help='Batch size for captioning')
def caption(input_dir, backend, batch_size):
    """Caption images in dataset."""
    from dataset.captioning.blip2 import BLIP2Captioner

    click.echo(f"Captioning images in {input_dir} with {backend}")

    if backend == 'blip2':
        captioner = BLIP2Captioner()
        captioner.load()

        # Find images
        image_exts = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [
            str(f) for f in Path(input_dir).iterdir()
            if f.suffix.lower() in image_exts
        ]

        click.echo(f"Found {len(image_files)} images")

        # Caption in batches
        captions = captioner.caption_batch(image_files, batch_size=batch_size)

        # Save captions
        for img_path, caption in zip(image_files, captions):
            caption_path = Path(img_path).with_suffix('.txt')
            caption_path.write_text(caption)

        click.echo(f"Captions saved")


# Training commands
@cli.group()
def train():
    """LoRA training commands."""
    pass


@train.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('output_name')
@click.option('--model', type=click.Choice(['flux', 'sdxl', 'sd3']), default='flux')
@click.option('--steps', default=1000, help='Training steps')
@click.option('--batch-size', default=1, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--rank', default=16, help='LoRA rank')
@click.option('--resolution', default=512, help='Training resolution')
@click.pass_context
def lora(ctx, dataset_dir, output_name, model, steps, batch_size, lr, rank, resolution):
    """Train LoRA on dataset."""
    config = ctx.obj['config']

    click.echo(f"Training {model} LoRA: {output_name}")
    click.echo(f"  Dataset: {dataset_dir}")
    click.echo(f"  Steps: {steps}")
    click.echo(f"  Batch size: {batch_size}")
    click.echo(f"  Learning rate: {lr}")
    click.echo(f"  Rank: {rank}")

    # Update config
    config.training.steps = steps
    config.training.batch_size = batch_size
    config.training.learning_rate = lr
    config.training.lora_rank = rank
    config.training.resolution = resolution

    click.echo("\n[Training implementation goes here]")
    click.echo("This would call the appropriate training backend based on model choice")


# Generation commands
@cli.group()
def generate():
    """Media generation commands."""
    pass


@generate.command()
@click.argument('prompt')
@click.option('--lora', help='Path to LoRA weights')
@click.option('--output', '-o', default='./output/images', help='Output directory')
@click.option('--num-images', '-n', default=4, help='Number of images to generate')
@click.option('--resolution', default=1024, help='Image resolution')
@click.option('--steps', default=30, help='Inference steps')
@click.option('--lora-strength', default=0.8, help='LoRA strength')
@click.option('--model', type=click.Choice(['flux', 'sdxl']), default='flux')
def image(prompt, lora, output, num_images, resolution, steps, lora_strength, model):
    """Generate images from text prompt."""
    click.echo(f"Generating {num_images} images with {model}")
    click.echo(f"Prompt: {prompt}")

    if model == 'flux':
        from generation.image.flux_gen import quick_generate

        paths = quick_generate(
            prompt=prompt,
            lora_path=lora,
            output_dir=output,
            num_images=num_images,
            resolution=resolution,
            steps=steps,
            lora_strength=lora_strength,
        )

        click.echo(f"\nGenerated {len(paths)} images:")
        for path in paths:
            click.echo(f"  {path}")


@generate.command()
@click.argument('prompt')
@click.option('--lora', help='Path to LoRA weights')
@click.option('--output', '-o', default='./output/video.mp4', help='Output video path')
@click.option('--num-frames', default=16, help='Number of frames')
@click.option('--resolution', default=512, help='Frame resolution')
@click.option('--fps', default=8, help='Frames per second')
@click.option('--lora-strength', default=0.8, help='LoRA strength')
def video(prompt, lora, output, num_frames, resolution, fps, lora_strength):
    """Generate video from text prompt."""
    from generation.video.animatediff import quick_video

    click.echo(f"Generating video with AnimateDiff")
    click.echo(f"Prompt: {prompt}")

    video_path = quick_video(
        prompt=prompt,
        lora_path=lora,
        output_path=output,
        num_frames=num_frames,
        resolution=resolution,
        fps=fps,
        lora_strength=lora_strength,
    )

    click.echo(f"\nVideo saved: {video_path}")


# LoRA utilities
@cli.group()
def lora():
    """LoRA management commands."""
    pass


@lora.command()
@click.argument('lora_paths', nargs=-1, required=True)
@click.argument('output_path')
@click.option('--weights', help='Comma-separated weights (e.g., 0.5,0.5)')
def merge(lora_paths, output_path, weights):
    """Merge multiple LoRAs."""
    click.echo(f"Merging {len(lora_paths)} LoRAs")
    click.echo(f"Output: {output_path}")

    if weights:
        weights = [float(w) for w in weights.split(',')]
    else:
        weights = [1.0 / len(lora_paths)] * len(lora_paths)

    click.echo(f"Weights: {weights}")
    click.echo("\n[LoRA merging implementation goes here]")


# Cloud commands
@cli.group()
def cloud():
    """Cloud compute management."""
    pass


@cloud.command()
@click.option('--gpu-type', default='RTX4090', help='GPU type')
@click.option('--region', default='US-OR', help='Region')
def provision(gpu_type, region):
    """Provision cloud GPU instance."""
    click.echo(f"Provisioning {gpu_type} in {region}")
    click.echo("\n[RunPod provisioning implementation goes here]")


@cloud.command()
def status():
    """Check cloud instance status."""
    click.echo("Checking cloud instance status...")
    click.echo("\n[RunPod status check goes here]")


if __name__ == '__main__':
    cli()
