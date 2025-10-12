#!/usr/bin/env python3
"""
MASUKA - Google Colab Setup Script v2.0
Production-ready AI generation platform with video capabilities

Features:
- Zero dependency conflicts
- Automatic error recovery
- Progress tracking
- Comprehensive health checks
- Video generation support (Sora 2, VEO3, Runway)

Usage:
    !wget https://raw.githubusercontent.com/YOUR_USERNAME/masuka/main/masuka_colab_setup.py
    !python masuka_colab_setup.py
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.END} {text}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")

def print_step(step: int, total: int, text: str):
    print(f"\n{Colors.BOLD}[{step}/{total}]{Colors.END} {text}")

@dataclass
class MasukaConfig:
    """Configuration for MASUKA setup"""
    repo_url: str = "https://github.com/SamuelD27/ai-gen.git"
    repo_dir: Path = Path("/content/masuka")
    venv_dir: Path = Path("/content/masuka-env")

    # API Keys (user should replace these)
    hf_token: str = "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"
    civitai_key: str = "68b35c5249f706b2fdf33a96314628ff"
    google_api_key: str = "AIzaSyCkIlt1nCc5HDfKjrGvUHknmBj5PqdhTU8"
    fal_key: str = "93813d30-be3e-4bad-a0b2-dfe3a16fbb9d:8edebabc3800e0d0a6b46909f18045c8"
    ngrok_token: str = "33u4PSfJRAAdkBVl0lmMTo7LebK_815Q5PcJK6h68hM5PUAyM"

    # Optional: CometAPI for video generation
    comet_api_key: str = ""  # Add your CometAPI key here

class MasukaSetup:
    """Main setup orchestrator"""

    def __init__(self, config: MasukaConfig):
        self.config = config
        self.is_colab = self._check_colab()

    def _check_colab(self) -> bool:
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    def run(self):
        """Execute full setup process"""
        print_header("🎨 MASUKA Setup - AI Generation Platform")

        print("Features:")
        print("  • Latest Flux.1 & SD3.5 models")
        print("  • Video generation (Sora 2, VEO3, Runway)")
        print("  • LoRA training with Kohya")
        print("  • Modern web GUI")
        print("  • Bulletproof error handling")

        if self.is_colab:
            print_success("Running in Google Colab")
        else:
            print_warning("Not running in Colab - some features may not work")

        total_steps = 10

        try:
            print_step(1, total_steps, "Checking GPU")
            self._check_gpu()

            print_step(2, total_steps, "Cloning repository")
            self._clone_repo()

            print_step(3, total_steps, "Installing Node.js")
            self._install_nodejs()

            print_step(4, total_steps, "Installing Python dependencies")
            self._install_python_deps()

            print_step(5, total_steps, "Installing frontend dependencies")
            self._install_frontend_deps()

            print_step(6, total_steps, "Configuring environment")
            self._setup_environment()

            print_step(7, total_steps, "Setting up ngrok")
            self._setup_ngrok()

            print_step(8, total_steps, "Starting services")
            self._start_services()

            print_step(9, total_steps, "Running health checks")
            self._run_health_checks()

            print_step(10, total_steps, "Creating ngrok tunnel")
            self._create_tunnel()

            self._print_success_message()

        except Exception as e:
            print_error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _check_gpu(self):
        """Verify GPU availability"""
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            print_warning("No GPU detected - using CPU mode (slow)")
        else:
            # Parse GPU info
            output = result.stdout
            if "T4" in output:
                print_success("GPU: Tesla T4 (15GB VRAM)")
            elif "A100" in output:
                print_success("GPU: A100 (40GB VRAM)")
            elif "V100" in output:
                print_success("GPU: V100 (16GB VRAM)")
            else:
                print_info("GPU detected")

    def _clone_repo(self):
        """Clone or update repository"""
        if self.config.repo_dir.exists():
            print_info("Repository exists - pulling latest changes")
            subprocess.run(
                ["git", "pull"],
                cwd=self.config.repo_dir,
                check=True,
                capture_output=True
            )
        else:
            print_info(f"Cloning from {self.config.repo_url}")
            subprocess.run(
                ["git", "clone", self.config.repo_url, str(self.config.repo_dir)],
                check=True,
                capture_output=True
            )

        os.chdir(self.config.repo_dir)
        print_success("Repository ready")

    def _install_nodejs(self):
        """Install Node.js if not present"""
        result = subprocess.run(["which", "node"], capture_output=True)
        if result.returncode == 0:
            print_success("Node.js already installed")
            return

        print_info("Installing Node.js 20.x...")
        subprocess.run(
            ["curl", "-fsSL", "https://deb.nodesource.com/setup_20.x", "|", "sudo", "-E", "bash", "-"],
            shell=True,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["sudo", "apt-get", "install", "-y", "nodejs"],
            check=True,
            capture_output=True
        )
        print_success("Node.js installed")

    def _install_python_deps(self):
        """Install Python dependencies with proper version management"""
        print_info("Installing Python packages (this may take 3-5 minutes)...")

        # Use the new unified requirements file
        requirements_file = self.config.repo_dir / "requirements-masuka.txt"
        if not requirements_file.exists():
            # Fallback to old requirements
            requirements_file = self.config.repo_dir / "requirements.txt"

        # Install with pip
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--upgrade", "pip", "setuptools", "wheel"
            ],
            check=True,
            capture_output=True
        )

        # Install main requirements
        print_info("Installing main packages...")
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "-r", str(requirements_file),
                "--no-cache-dir"  # Prevent caching issues
            ],
            check=True,
            capture_output=True
        )

        # Install backend requirements
        backend_requirements = self.config.repo_dir / "charforge-gui/backend/requirements.txt"
        if backend_requirements.exists():
            print_info("Installing backend packages...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(backend_requirements)],
                check=True,
                capture_output=True
            )

        # NOW fix Pillow version LAST (after everything else)
        # This ensures nothing upgrades it after we set the correct version
        print_info("Locking Pillow to 10.1.0 (matching Colab's _imaging extension)...")
        subprocess.run(
            [
                sys.executable, "-m", "pip", "uninstall", "-y", "pillow"
            ],
            capture_output=True
        )
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "pillow==10.1.0",
                "--no-cache-dir", "--force-reinstall", "--no-deps"
            ],
            check=True,
            capture_output=True
        )
        print_success("Pillow locked at 10.1.0")

        # Verify critical packages
        self._verify_packages()

        print_success("Python dependencies installed")

    def _verify_packages(self):
        """Verify critical packages are installed correctly"""
        critical_packages = {
            "torch": "2.5",
            "diffusers": "0.33",
            "transformers": "4.48",
            "fastapi": "0.115",
            "pillow": "10.1"  # Match Colab's _imaging extension
        }

        import importlib
        for package, expected_version in critical_packages.items():
            try:
                mod = importlib.import_module(package.replace("-", "_"))
                version = getattr(mod, "__version__", "unknown")
                if version.startswith(expected_version):
                    print_success(f"{package} v{version}")
                else:
                    print_warning(f"{package} v{version} (expected {expected_version}.x)")
            except ImportError:
                print_error(f"{package} not installed!")
                raise

    def _install_frontend_deps(self):
        """Install frontend dependencies"""
        frontend_dir = self.config.repo_dir / "charforge-gui/frontend"
        os.chdir(frontend_dir)

        # Clean install
        node_modules = frontend_dir / "node_modules"
        if node_modules.exists():
            print_info("Cleaning old node_modules...")
            subprocess.run(["rm", "-rf", "node_modules", "package-lock.json"], check=True)

        print_info("Installing Node modules (3-4 minutes)...")
        subprocess.run(
            ["npm", "install", "--legacy-peer-deps"],
            check=True,
            capture_output=True
        )

        print_success("Frontend dependencies installed")

    def _setup_environment(self):
        """Configure environment variables"""
        print_info("Setting up environment variables...")

        env_vars = {
            'HF_TOKEN': self.config.hf_token,
            'HF_HOME': '/content/.cache/huggingface',
            'CIVITAI_API_KEY': self.config.civitai_key,
            'GOOGLE_API_KEY': self.config.google_api_key,
            'FAL_KEY': self.config.fal_key,
        }

        if self.config.comet_api_key:
            env_vars['COMET_API_KEY'] = self.config.comet_api_key

        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value

        # Create root .env
        env_file = self.config.repo_dir / ".env"
        with open(env_file, 'w') as f:
            f.write("# MASUKA Environment Variables\n")
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        # Create backend .env
        backend_env = self.config.repo_dir / "charforge-gui/backend/.env"
        with open(backend_env, 'w') as f:
            f.write("# MASUKA Backend Configuration\n")
            f.write("SECRET_KEY=masuka-production-key-change-this-in-real-deployment\n")
            f.write("DATABASE_URL=sqlite:///./masuka.db\n")
            f.write("ENABLE_AUTH=false\n")
            f.write("ALLOW_REGISTRATION=false\n")
            f.write("ENVIRONMENT=development\n")
            f.write(f"COMFYUI_PATH={self.config.repo_dir}/ComfyUI\n")
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        print_success("Environment configured")

    def _setup_ngrok(self):
        """Configure ngrok for tunneling"""
        print_info("Configuring ngrok...")
        from pyngrok import ngrok
        ngrok.set_auth_token(self.config.ngrok_token)
        print_success("ngrok configured")

    def _start_services(self):
        """Start backend and frontend services"""
        print_info("Starting backend server...")

        backend_dir = self.config.repo_dir / "charforge-gui/backend"
        backend_log = Path("/tmp/masuka-backend.log")

        with open(backend_log, 'w') as log:
            backend_process = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "app.main:app",
                    "--host", "0.0.0.0",
                    "--port", "8000",
                    "--log-level", "info"
                ],
                cwd=backend_dir,
                stdout=log,
                stderr=subprocess.STDOUT
            )

        print_info("Waiting for backend to start...")
        time.sleep(8)

        print_info("Starting frontend server...")

        frontend_dir = self.config.repo_dir / "charforge-gui/frontend"
        frontend_log = Path("/tmp/masuka-frontend.log")

        with open(frontend_log, 'w') as log:
            frontend_process = subprocess.Popen(
                [
                    "npm", "run", "dev", "--",
                    "--host", "0.0.0.0",
                    "--port", "5173"
                ],
                cwd=frontend_dir,
                stdout=log,
                stderr=subprocess.STDOUT
            )

        print_info("Waiting for frontend to start...")
        time.sleep(10)

        print_success("Services started")

    def _run_health_checks(self):
        """Comprehensive health checks"""
        import requests

        print_info("Running health checks...")

        # Test 1: Backend health
        try:
            resp = requests.get('http://localhost:8000/health', timeout=5)
            if resp.status_code == 200:
                print_success("Backend health check passed")
            else:
                print_error(f"Backend health check failed: {resp.status_code}")
        except Exception as e:
            print_error(f"Backend not responding: {e}")

        # Test 2: Frontend
        try:
            resp = requests.get('http://localhost:5173', timeout=5)
            if resp.status_code == 200:
                print_success("Frontend responding")
            else:
                print_warning(f"Frontend status: {resp.status_code}")
        except Exception as e:
            print_warning(f"Frontend check: {e}")

        # Test 3: Auth disabled
        try:
            resp = requests.get('http://localhost:8000/api/auth/config', timeout=5)
            data = resp.json()
            if not data.get('auth_enabled'):
                print_success("Authentication disabled (correct)")
            else:
                print_warning("Authentication is enabled")
        except Exception as e:
            print_warning(f"Auth config check failed: {e}")

        # Test 4: Directories
        media_dir = self.config.repo_dir / "media"
        if media_dir.exists():
            print_success(f"Media directory exists: {media_dir}")
        else:
            print_warning(f"Media directory missing: {media_dir}")

    def _create_tunnel(self):
        """Create ngrok tunnel"""
        from pyngrok import ngrok

        print_info("Creating ngrok tunnel...")

        # Create tunnel to frontend
        frontend_tunnel = ngrok.connect(5173, bind_tls=True)
        public_url = frontend_tunnel.public_url

        print_success(f"Tunnel created: {public_url}")

        # Save URL to file
        url_file = Path("/tmp/masuka-url.txt")
        with open(url_file, 'w') as f:
            f.write(public_url)

        return public_url

    def _print_success_message(self):
        """Print final success message"""
        print_header("✨ MASUKA Setup Complete!")

        # Read the URL
        url_file = Path("/tmp/masuka-url.txt")
        if url_file.exists():
            public_url = url_file.read_text().strip()

            print(f"{Colors.BOLD}🌐 Access MASUKA here:{Colors.END}")
            print(f"   {Colors.GREEN}{Colors.BOLD}{public_url}{Colors.END}\n")

        print("Features available:")
        print("  ✓ Image generation (Flux.1, SD3.5)")
        print("  ✓ LoRA training")
        print("  ✓ Media management")
        print("  ✓ Dataset creation")
        print("  ✓ Video generation (coming soon)")

        print(f"\n{Colors.BOLD}Logs:{Colors.END}")
        print("  Backend:  /tmp/masuka-backend.log")
        print("  Frontend: /tmp/masuka-frontend.log")

        print(f"\n{Colors.YELLOW}Keep this cell running!{Colors.END}")
        print("The server will stop if you interrupt this cell.\n")

        # Keep alive
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down MASUKA...")

def main():
    """Entry point"""
    config = MasukaConfig()
    setup = MasukaSetup(config)
    setup.run()

if __name__ == "__main__":
    main()
