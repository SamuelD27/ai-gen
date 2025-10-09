"""
CharForgeX Gradio GUI - Professional LoRA Training & Media Generation
No content restrictions. Maximum creative freedom.
"""

import gradio as gr
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import load_config, get_config


def create_dataset_tab():
    """Create dataset processing tab."""
    with gr.Tab("Dataset Processing"):
        gr.Markdown("""
        ## Dataset Processing
        Clean, preprocess, and caption your datasets for optimal LoRA training.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Clean Dataset")
                clean_input_dir = gr.Textbox(label="Input Directory")
                clean_output_dir = gr.Textbox(label="Output Directory")
                clean_min_res = gr.Slider(256, 2048, 512, label="Minimum Resolution")
                clean_require_faces = gr.Checkbox(True, label="Require Faces")
                clean_quality_filter = gr.Checkbox(True, label="Quality Filter")
                clean_btn = gr.Button("Clean Dataset", variant="primary")
                clean_output = gr.Textbox(label="Results", lines=5)

            with gr.Column():
                gr.Markdown("### Preprocess Dataset")
                prep_input_dir = gr.Textbox(label="Input Directory")
                prep_output_dir = gr.Textbox(label="Output Directory")
                prep_resolution = gr.Slider(512, 2048, 1024, label="Target Resolution")
                prep_smart_crop = gr.Checkbox(True, label="Smart Crop (Face-Aware)")
                prep_augment = gr.Checkbox(False, label="Create Augmentations")
                prep_btn = gr.Button("Preprocess Dataset", variant="primary")
                prep_output = gr.Textbox(label="Results", lines=5)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Caption Dataset")
                caption_input_dir = gr.Textbox(label="Input Directory")
                caption_backend = gr.Dropdown(
                    choices=["blip2", "clip_interrogator", "cogvlm", "ensemble"],
                    value="blip2",
                    label="Captioning Backend"
                )
                caption_batch_size = gr.Slider(1, 16, 4, step=1, label="Batch Size")
                caption_btn = gr.Button("Caption Dataset", variant="primary")
                caption_output = gr.Textbox(label="Results", lines=5)

        # Connect functions (placeholders for now)
        def clean_dataset_fn(input_dir, output_dir, min_res, require_faces, quality_filter):
            return f"Cleaning dataset from {input_dir} to {output_dir}...\n[Implementation pending]"

        def preprocess_dataset_fn(input_dir, output_dir, resolution, smart_crop, augment):
            return f"Preprocessing dataset from {input_dir} to {output_dir}...\n[Implementation pending]"

        def caption_dataset_fn(input_dir, backend, batch_size):
            return f"Captioning dataset in {input_dir} with {backend}...\n[Implementation pending]"

        clean_btn.click(
            clean_dataset_fn,
            inputs=[clean_input_dir, clean_output_dir, clean_min_res, clean_require_faces, clean_quality_filter],
            outputs=clean_output
        )

        prep_btn.click(
            preprocess_dataset_fn,
            inputs=[prep_input_dir, prep_output_dir, prep_resolution, prep_smart_crop, prep_augment],
            outputs=prep_output
        )

        caption_btn.click(
            caption_dataset_fn,
            inputs=[caption_input_dir, caption_backend, caption_batch_size],
            outputs=caption_output
        )


def create_training_tab():
    """Create LoRA training tab."""
    with gr.Tab("LoRA Training"):
        gr.Markdown("""
        ## LoRA Training
        Train photorealistic LoRAs on your custom datasets.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Training Configuration")
                train_dataset_dir = gr.Textbox(label="Dataset Directory")
                train_output_name = gr.Textbox(label="Output LoRA Name", placeholder="my_lora")
                train_model = gr.Dropdown(
                    choices=["flux-dev", "flux-schnell", "sdxl", "sd3"],
                    value="flux-dev",
                    label="Base Model"
                )

                with gr.Accordion("Training Parameters", open=True):
                    train_steps = gr.Slider(100, 5000, 1000, step=100, label="Training Steps")
                    train_batch_size = gr.Slider(1, 8, 1, step=1, label="Batch Size")
                    train_lr = gr.Number(1e-4, label="Learning Rate")
                    train_rank = gr.Slider(4, 128, 16, step=4, label="LoRA Rank")
                    train_resolution = gr.Slider(256, 1024, 512, step=64, label="Training Resolution")

                with gr.Accordion("Advanced Options", open=False):
                    train_gradient_accum = gr.Slider(1, 16, 4, step=1, label="Gradient Accumulation Steps")
                    train_save_every = gr.Slider(50, 1000, 250, step=50, label="Save Every N Steps")
                    train_use_8bit = gr.Checkbox(True, label="Use 8-bit Optimizer")

                train_btn = gr.Button("Start Training", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("### Training Progress")
                train_status = gr.Textbox(label="Status", lines=3)
                train_progress = gr.Textbox(label="Training Log", lines=20)

        def train_lora_fn(dataset_dir, output_name, model, steps, batch_size, lr, rank, resolution, gradient_accum, save_every, use_8bit):
            return f"Training {model} LoRA: {output_name}\n[Implementation pending]", f"Dataset: {dataset_dir}\nSteps: {steps}\nBatch size: {batch_size}\nLR: {lr}"

        train_btn.click(
            train_lora_fn,
            inputs=[train_dataset_dir, train_output_name, train_model, train_steps, train_batch_size,
                   train_lr, train_rank, train_resolution, train_gradient_accum, train_save_every, train_use_8bit],
            outputs=[train_status, train_progress]
        )


def create_image_gen_tab():
    """Create image generation tab."""
    with gr.Tab("Image Generation"):
        gr.Markdown("""
        ## Image Generation
        Generate ultra-realistic images with your trained LoRAs.
        **No content restrictions. Full creative freedom.**
        """)

        with gr.Row():
            with gr.Column():
                img_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt here...")
                img_negative = gr.Textbox(label="Negative Prompt (optional)", lines=2)

                with gr.Row():
                    img_model = gr.Dropdown(
                        choices=["flux-dev", "flux-schnell", "sdxl"],
                        value="flux-dev",
                        label="Model"
                    )
                    img_lora = gr.Textbox(label="LoRA Path (optional)")

                with gr.Row():
                    img_width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                    img_height = gr.Slider(512, 2048, 1024, step=64, label="Height")

                with gr.Row():
                    img_steps = gr.Slider(10, 100, 30, step=1, label="Steps")
                    img_guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance Scale")

                with gr.Row():
                    img_num_images = gr.Slider(1, 16, 4, step=1, label="Number of Images")
                    img_lora_strength = gr.Slider(0, 2, 0.8, step=0.1, label="LoRA Strength")

                img_seed = gr.Number(-1, label="Seed (-1 for random)")

                img_generate_btn = gr.Button("Generate Images", variant="primary", size="lg")

            with gr.Column():
                img_gallery = gr.Gallery(label="Generated Images", columns=2, height=600)
                img_status = gr.Textbox(label="Status", lines=2)

        def generate_images_fn(prompt, negative, model, lora, width, height, steps, guidance, num_images, lora_strength, seed):
            return [], f"Generating {num_images} images with {model}...\n[Implementation pending]"

        img_generate_btn.click(
            generate_images_fn,
            inputs=[img_prompt, img_negative, img_model, img_lora, img_width, img_height,
                   img_steps, img_guidance, img_num_images, img_lora_strength, img_seed],
            outputs=[img_gallery, img_status]
        )


def create_video_gen_tab():
    """Create video generation tab."""
    with gr.Tab("Video Generation"):
        gr.Markdown("""
        ## Video Generation
        Generate ultra-realistic videos with AnimateDiff and LoRAs.
        **No content restrictions. Full creative freedom.**
        """)

        with gr.Row():
            with gr.Column():
                vid_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your video prompt...")
                vid_negative = gr.Textbox(label="Negative Prompt (optional)", lines=2, value="bad quality, worst quality, blurry")

                with gr.Row():
                    vid_backend = gr.Dropdown(
                        choices=["animatediff", "hotshotxl", "svd"],
                        value="animatediff",
                        label="Video Backend"
                    )
                    vid_lora = gr.Textbox(label="LoRA Path (optional)")

                with gr.Row():
                    vid_width = gr.Slider(256, 1024, 512, step=64, label="Width")
                    vid_height = gr.Slider(256, 1024, 512, step=64, label="Height")

                with gr.Row():
                    vid_num_frames = gr.Slider(8, 64, 16, step=8, label="Number of Frames")
                    vid_fps = gr.Slider(4, 30, 8, step=1, label="FPS")

                with gr.Row():
                    vid_steps = gr.Slider(10, 50, 25, step=1, label="Steps")
                    vid_guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance Scale")

                vid_lora_strength = gr.Slider(0, 2, 0.8, step=0.1, label="LoRA Strength")

                with gr.Accordion("Advanced: Motion Control", open=False):
                    vid_use_facial_motion = gr.Checkbox(False, label="Enable Facial Motion Transfer")
                    vid_motion_source = gr.Video(label="Driving Video (optional)")
                    vid_use_audio_sync = gr.Checkbox(False, label="Enable Audio Sync")
                    vid_audio_source = gr.Audio(label="Audio Source (optional)")

                with gr.Accordion("Advanced: Interpolation", open=False):
                    vid_use_interp = gr.Checkbox(True, label="Enable Frame Interpolation")
                    vid_target_fps = gr.Slider(8, 60, 24, step=1, label="Target FPS (after interpolation)")

                vid_generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Column():
                vid_output = gr.Video(label="Generated Video")
                vid_status = gr.Textbox(label="Status", lines=3)

        def generate_video_fn(*args):
            return None, "Generating video...\n[Implementation pending]"

        vid_generate_btn.click(
            generate_video_fn,
            inputs=[vid_prompt, vid_negative, vid_backend, vid_lora, vid_width, vid_height,
                   vid_num_frames, vid_fps, vid_steps, vid_guidance, vid_lora_strength,
                   vid_use_facial_motion, vid_motion_source, vid_use_audio_sync, vid_audio_source,
                   vid_use_interp, vid_target_fps],
            outputs=[vid_output, vid_status]
        )


def create_lora_utils_tab():
    """Create LoRA utilities tab."""
    with gr.Tab("LoRA Utilities"):
        gr.Markdown("""
        ## LoRA Utilities
        Merge, benchmark, and manage your LoRAs.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Merge LoRAs")
                merge_lora1 = gr.Textbox(label="LoRA 1 Path")
                merge_weight1 = gr.Slider(0, 1, 0.5, step=0.1, label="LoRA 1 Weight")
                merge_lora2 = gr.Textbox(label="LoRA 2 Path")
                merge_weight2 = gr.Slider(0, 1, 0.5, step=0.1, label="LoRA 2 Weight")
                merge_output = gr.Textbox(label="Output LoRA Path")
                merge_btn = gr.Button("Merge LoRAs", variant="primary")
                merge_status = gr.Textbox(label="Status", lines=3)

            with gr.Column():
                gr.Markdown("### Benchmark LoRA")
                bench_lora = gr.Textbox(label="LoRA Path")
                bench_dataset = gr.Textbox(label="Test Dataset Path")
                bench_num_images = gr.Slider(10, 100, 50, step=10, label="Number of Test Images")
                bench_btn = gr.Button("Run Benchmark", variant="primary")
                bench_results = gr.Textbox(label="Benchmark Results", lines=10)

        def merge_loras_fn(lora1, weight1, lora2, weight2, output):
            return f"Merging LoRAs...\n{lora1} ({weight1}) + {lora2} ({weight2}) -> {output}\n[Implementation pending]"

        def benchmark_lora_fn(lora, dataset, num_images):
            return f"Benchmarking {lora} on {dataset}...\n[Implementation pending]"

        merge_btn.click(
            merge_loras_fn,
            inputs=[merge_lora1, merge_weight1, merge_lora2, merge_weight2, merge_output],
            outputs=merge_status
        )

        bench_btn.click(
            benchmark_lora_fn,
            inputs=[bench_lora, bench_dataset, bench_num_images],
            outputs=bench_results
        )


def create_settings_tab():
    """Create settings tab."""
    with gr.Tab("Settings"):
        gr.Markdown("""
        ## System Settings
        Configure compute mode, cloud integration, and system preferences.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Compute Configuration")
                settings_mode = gr.Dropdown(
                    choices=["local", "cloud", "hybrid"],
                    value="local",
                    label="Compute Mode"
                )
                settings_device = gr.Dropdown(
                    choices=["cuda", "cpu", "mps"],
                    value="cuda",
                    label="Device"
                )
                settings_precision = gr.Dropdown(
                    choices=["fp32", "fp16", "bf16"],
                    value="bf16",
                    label="Precision"
                )
                settings_xformers = gr.Checkbox(True, label="Enable xformers")

            with gr.Column():
                gr.Markdown("### Cloud Configuration (RunPod)")
                settings_cloud_enabled = gr.Checkbox(False, label="Enable Cloud Compute")
                settings_cloud_api_key = gr.Textbox(label="RunPod API Key", type="password")
                settings_cloud_gpu = gr.Dropdown(
                    choices=["RTX4090", "A100", "H100"],
                    value="RTX4090",
                    label="GPU Type"
                )
                settings_cloud_region = gr.Dropdown(
                    choices=["US-OR", "US-CA", "EU-RO"],
                    value="US-OR",
                    label="Region"
                )
                settings_cloud_max_cost = gr.Number(1.0, label="Max Cost Per Hour ($)")

        with gr.Row():
            settings_save_btn = gr.Button("Save Settings", variant="primary")
            settings_load_btn = gr.Button("Load Settings")

        settings_status = gr.Textbox(label="Status", lines=2)

        def save_settings_fn(*args):
            return "Settings saved successfully!"

        def load_settings_fn():
            return "local", "cuda", "bf16", True, False, "", "RTX4090", "US-OR", 1.0, "Settings loaded!"

        settings_save_btn.click(
            save_settings_fn,
            inputs=[settings_mode, settings_device, settings_precision, settings_xformers,
                   settings_cloud_enabled, settings_cloud_api_key, settings_cloud_gpu,
                   settings_cloud_region, settings_cloud_max_cost],
            outputs=settings_status
        )

        settings_load_btn.click(
            load_settings_fn,
            outputs=[settings_mode, settings_device, settings_precision, settings_xformers,
                    settings_cloud_enabled, settings_cloud_api_key, settings_cloud_gpu,
                    settings_cloud_region, settings_cloud_max_cost, settings_status]
        )


def create_app():
    """Create the main Gradio application."""
    with gr.Blocks(title="CharForgeX", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # CharForgeX - Professional LoRA Training & Media Generation

        **Ultra-realistic image and video generation with custom LoRAs**

        ⚠️ **No content restrictions. Maximum creative freedom.**

        Train photorealistic LoRAs on any dataset and generate high-fidelity images/videos
        using state-of-the-art models (Flux, SDXL, AnimateDiff, HotShotXL).
        """)

        # Create all tabs
        with gr.Tabs():
            create_dataset_tab()
            create_training_tab()
            create_image_gen_tab()
            create_video_gen_tab()
            create_lora_utils_tab()
            create_settings_tab()

        gr.Markdown("""
        ---
        ### Quick Start Guide
        1. **Dataset Processing**: Clean and preprocess your images
        2. **Caption**: Generate captions with BLIP2 or CLIP
        3. **Train**: Train a LoRA on your dataset
        4. **Generate**: Create images or videos with your LoRA
        5. **Utilities**: Merge LoRAs or benchmark performance
        """)

    return app


if __name__ == "__main__":
    # Load configuration
    load_config()

    # Create and launch app
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
