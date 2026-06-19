#!/bin/bash
# Scale-MAE Complete Installation Script with Error Checking
# This script installs everything step-by-step and validates each step

set -e  # Exit on any error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================"
echo "Scale-MAE Complete Installation"
echo "======================================${NC}"
echo ""

# Function to print step
print_step() {
    echo -e "\n${BLUE}[STEP $1/$2]${NC} $3"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error and exit
print_error() {
    echo -e "${RED}✗ ERROR:${NC} $1"
    echo "Installation failed. Please fix the error and try again."
    exit 1
}

TOTAL_STEPS=12

# ============ STEP 1: Check Python version ============
print_step 1 $TOTAL_STEPS "Checking Python version"
if ! command -v python &> /dev/null; then
    print_error "Python not found. Load Python module: module load python/3.9"
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 8 ] || [ "$PYTHON_MINOR" -gt 10 ]; then
    print_error "Python version $PYTHON_VERSION not supported. Need Python 3.8-3.10"
fi
print_success "Python $PYTHON_VERSION found"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/scalemae_env"
REPO_DIR="$SCRIPT_DIR/scale-mae"

echo "Working directory: $SCRIPT_DIR"
echo "Virtual environment will be: $VENV_DIR"
echo "Repository will be: $REPO_DIR"

# ============ STEP 2: Create virtual environment ============
print_step 2 $TOTAL_STEPS "Creating virtual environment"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf "$VENV_DIR"
fi

python -m venv "$VENV_DIR" || print_error "Failed to create virtual environment"
print_success "Virtual environment created at $VENV_DIR"

# ============ STEP 3: Activate virtual environment ============
print_step 3 $TOTAL_STEPS "Activating virtual environment"
source "$VENV_DIR/bin/activate" || print_error "Failed to activate virtual environment"
print_success "Virtual environment activated"

# ============ STEP 4: Upgrade pip ============
print_step 4 $TOTAL_STEPS "Upgrading pip"
pip install --upgrade pip setuptools wheel || print_error "Failed to upgrade pip"
pip --version
print_success "pip upgraded"

# ============ STEP 5: Check and install GDAL ============
print_step 5 $TOTAL_STEPS "Installing GDAL"
echo "Checking for system GDAL..."

# First, install numpy (GDAL needs it)
echo "Installing numpy (required for GDAL)..."
pip install "numpy>=1.21.0,<2.0.0" || print_error "Failed to install numpy"
print_success "numpy installed"

GDAL_INSTALLED=false

# Try to find gdal-config
if command -v gdal-config &> /dev/null; then
    GDAL_VERSION=$(gdal-config --version)
    echo "System GDAL version: $GDAL_VERSION"
    
    # Check if version is too old (< 3.1.0 has setuptools issues)
    GDAL_MAJOR=$(echo $GDAL_VERSION | cut -d. -f1)
    GDAL_MINOR=$(echo $GDAL_VERSION | cut -d. -f2)
    
    if [ "$GDAL_MAJOR" -lt 3 ] || ([ "$GDAL_MAJOR" -eq 3 ] && [ "$GDAL_MINOR" -lt 1 ]); then
        echo "Warning: System GDAL $GDAL_VERSION is too old (has setuptools compatibility issues)"
        echo "Trying to install newer GDAL from pip instead..."
        
        # Try installing GDAL >= 3.4 from pip
        if pip install "GDAL>=3.4.0" 2>/dev/null; then
            print_success "Installed newer GDAL from pip"
            GDAL_INSTALLED=true
        elif pip install GDAL 2>/dev/null; then
            print_success "Installed GDAL from pip (latest available)"
            GDAL_INSTALLED=true
        else
            echo "Warning: Could not install GDAL from pip either"
        fi
    else
        echo "Installing Python GDAL bindings for version $GDAL_VERSION..."
        if pip install "GDAL==$GDAL_VERSION" 2>/dev/null; then
            print_success "GDAL Python bindings installed"
            GDAL_INSTALLED=true
        else
            echo "Warning: Failed to install GDAL $GDAL_VERSION from pip"
            echo "Trying without version constraint..."
            if pip install GDAL 2>/dev/null; then
                print_success "Installed GDAL from pip (may be different version)"
                GDAL_INSTALLED=true
            fi
        fi
    fi
else
    echo "System GDAL not found (gdal-config not in PATH)"
    echo "Trying to load via module system..."
    
    # Try common module names
    for gdal_module in gdal/3.6.0 gdal/3.5.0 gdal/3.4.1 GDAL/3.6.0 GDAL/3.5.0 GDAL/3.4.1 gdal GDAL; do
        if module load $gdal_module 2>/dev/null; then
            echo "Loaded module: $gdal_module"
            if command -v gdal-config &> /dev/null; then
                GDAL_VERSION=$(gdal-config --version)
                echo "Module provides GDAL version: $GDAL_VERSION"
                
                if pip install "GDAL==$GDAL_VERSION" 2>/dev/null; then
                    print_success "GDAL installed via module"
                    GDAL_INSTALLED=true
                    break
                else
                    echo "Failed to install Python bindings for module GDAL version"
                    module unload $gdal_module 2>/dev/null
                fi
            fi
        fi
    done
    
    # If still not installed, try pip without system GDAL
    if [ "$GDAL_INSTALLED" = false ]; then
        echo "No compatible system GDAL found"
        echo "Attempting pip installation of GDAL (may work without system GDAL)..."
        
        if pip install GDAL 2>/dev/null; then
            print_success "GDAL installed from pip"
            GDAL_INSTALLED=true
        else
            echo "Warning: Could not install GDAL"
        fi
    fi
fi

# Verify GDAL installation
if [ "$GDAL_INSTALLED" = true ]; then
    if python -c "from osgeo import gdal; print(f'GDAL {gdal.__version__} installed')" 2>/dev/null; then
        print_success "GDAL verified and working"
    else
        echo "Warning: GDAL installed but import failed"
        GDAL_INSTALLED=false
    fi
fi

# If GDAL still not working, continue anyway with warning
if [ "$GDAL_INSTALLED" = false ]; then
    echo ""
    echo -e "${YELLOW}============================================${NC}"
    echo -e "${YELLOW}WARNING: GDAL installation failed${NC}"
    echo -e "${YELLOW}============================================${NC}"
    echo "This is OK for basic EuroSAT testing, as it uses standard image formats."
    echo "GDAL is only needed for some geospatial data formats (GeoTIFF, etc.)"
    echo ""
    echo "Continuing installation without GDAL..."
    echo "You can try to install it manually later if needed."
    echo ""
    # Don't exit - continue installation
fi

# ============ STEP 6: Install geospatial libraries ============
print_step 6 $TOTAL_STEPS "Installing geospatial libraries"

# These may fail if GDAL isn't installed, but try anyway
echo "Attempting to install rasterio, shapely, geopandas..."

GEOSPATIAL_SUCCESS=false

if pip install rasterio shapely geopandas 2>/dev/null; then
    print_success "Geospatial libraries installed"
    GEOSPATIAL_SUCCESS=true
    
    # Verify installations
    python -c "import rasterio" 2>/dev/null && echo "  ✓ rasterio working"
    python -c "import shapely" 2>/dev/null && echo "  ✓ shapely working"
    python -c "import geopandas" 2>/dev/null && echo "  ✓ geopandas working"
else
    echo "Warning: Some geospatial libraries failed to install"
    echo "Trying to install individually..."
    
    # Try shapely first (doesn't need GDAL)
    if pip install shapely 2>/dev/null; then
        echo "  ✓ shapely installed"
    fi
    
    # Try rasterio (may need GDAL)
    if pip install rasterio 2>/dev/null; then
        echo "  ✓ rasterio installed"
    else
        echo "  ✗ rasterio failed (needs GDAL)"
    fi
    
    # Try geopandas (may need GDAL)
    if pip install geopandas 2>/dev/null; then
        echo "  ✓ geopandas installed"
    else
        echo "  ✗ geopandas failed (needs GDAL)"
    fi
fi

echo ""
echo -e "${YELLOW}Note: Geospatial library warnings are OK for basic EuroSAT testing${NC}"
echo "Scale-MAE will still work with standard image formats (JPG/PNG)"
echo ""

# ============ STEP 7: Check CUDA and install PyTorch ============
print_step 7 $TOTAL_STEPS "Installing PyTorch with CUDA support"

# Check if CUDA module is loaded
CUDA_AVAILABLE=false
CUDA_VERSION=""

echo "Checking for CUDA..."

# Method 1: Check if CUDA module is loaded (improved parsing)
if module list 2>&1 | grep -i "cuda/" > /dev/null; then
    echo "✓ CUDA module is loaded"
    # Extract version from module name - look specifically for cuda/ pattern
    CUDA_MODULE=$(module list 2>&1 | grep -i "cuda/" | head -1)
    echo "  Found module: $CUDA_MODULE"
    # Extract just the version number after cuda/
    CUDA_VERSION=$(echo "$CUDA_MODULE" | sed -n 's/.*cuda\/\([0-9]\+\.[0-9]\+\).*/\1/p')
    if [ -n "$CUDA_VERSION" ]; then
        echo "  Extracted CUDA version: $CUDA_VERSION"
        CUDA_AVAILABLE=true
    fi
fi

# Method 2: Check for nvcc compiler (works on login nodes)
if [ "$CUDA_AVAILABLE" = false ] && command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    if [ -n "$CUDA_VERSION" ]; then
        echo "✓ CUDA detected via nvcc: $CUDA_VERSION"
        CUDA_AVAILABLE=true
    fi
fi

# Method 3: Check CUDA environment variables
if [ "$CUDA_AVAILABLE" = false ] && [ -n "$CUDA_HOME" ]; then
    CUDA_VERSION=$(basename "$CUDA_HOME" | grep -oP '\d+\.\d+')
    if [ -n "$CUDA_VERSION" ]; then
        echo "✓ CUDA detected via CUDA_HOME: $CUDA_VERSION"
        CUDA_AVAILABLE=true
    fi
fi

# Determine PyTorch wheel based on CUDA version
if [ "$CUDA_AVAILABLE" = true ] && [ -n "$CUDA_VERSION" ]; then
    echo "Using CUDA version: $CUDA_VERSION"
    
    # Map CUDA version to PyTorch wheel
    if [[ "$CUDA_VERSION" == "11.4" ]]; then
        TORCH_WHEEL="cu116"  # PyTorch doesn't have cu114, use cu116
    elif [[ "$CUDA_VERSION" == "11.6" ]]; then
        TORCH_WHEEL="cu116"
    elif [[ "$CUDA_VERSION" == "11.7" ]]; then
        TORCH_WHEEL="cu117"
    elif [[ "$CUDA_VERSION" == "11.8" ]]; then
        TORCH_WHEEL="cu118"
    elif [[ "$CUDA_VERSION" == "12.1" ]]; then
        TORCH_WHEEL="cu121"
    elif [[ "$CUDA_VERSION" == "12.4" ]] || [[ "$CUDA_VERSION" == "12.8" ]]; then
        TORCH_WHEEL="cu121"  # Use cu121 for CUDA 12.x
    else
        echo "Warning: CUDA version $CUDA_VERSION not directly supported. Using cu118"
        TORCH_WHEEL="cu118"
    fi
    
    print_success "CUDA $CUDA_VERSION detected, will install PyTorch with $TORCH_WHEEL"
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}ERROR: CUDA IS NOT AVAILABLE${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo "CUDA module is not loaded!"
    echo ""
    echo "Available CUDA modules on your cluster:"
    module avail cuda 2>&1 | grep "cuda/"
    echo ""
    echo "To fix:"
    echo "  1. Load CUDA module: module load cuda/12.1"
    echo "  2. Verify module loaded: module list | grep cuda"
    echo "  3. Delete venv: rm -rf scalemae_env"
    echo "  4. Re-run script: bash install_scalemae.sh"
    echo ""
    print_error "CUDA module not loaded. Please load CUDA and try again."
fi

echo "Installing PyTorch for CUDA $TORCH_WHEEL..."
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_WHEEL" || print_error "Failed to install PyTorch"

# Verify PyTorch installation (without nvidia-smi requirement)
echo "Verifying PyTorch installation..."
python << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available in PyTorch: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version in PyTorch: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print("✓ PyTorch CUDA support confirmed")
    print("")
    print("Note: GPU count will show 0 on login nodes - this is normal!")
    print("GPU will be available when you run on compute nodes via SLURM")
else:
    print("")
    print("=" * 60)
    print("WARNING: CUDA is not available in PyTorch!")
    print("=" * 60)
    print("")
    print("This usually means the CUDA module wasn't loaded properly.")
    print("")
    print("To fix:")
    print("  1. Load CUDA: module load cuda/12.1")
    print("  2. Check: module list | grep cuda")
    print("  3. Delete venv: rm -rf scalemae_env")
    print("  4. Re-run: bash install_scalemae.sh")
    print("")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    print_error "PyTorch CUDA verification failed"
fi

print_success "PyTorch with CUDA support installed"
echo ""
echo -e "${YELLOW}Note: GPU testing will happen on compute nodes, not login nodes${NC}"
echo ""

# ============ STEP 8: Install timm 0.3.2 ============
print_step 8 $TOTAL_STEPS "Installing timm 0.3.2 (critical version)"
pip install timm==0.3.2 || print_error "Failed to install timm"

# Verify timm version
python << 'EOF' || print_error "timm verification failed"
import timm
if timm.__version__ != "0.3.2":
    raise RuntimeError(f"Wrong timm version: {timm.__version__}. Need 0.3.2")
print(f"timm version: {timm.__version__}")
EOF

print_success "timm 0.3.2 installed"

# ============ STEP 9: Install other dependencies ============
print_step 9 $TOTAL_STEPS "Installing other dependencies"

# Core dependencies
echo "Installing core dependencies..."
pip install numpy pyyaml pillow || print_error "Failed to install core dependencies"

# Visualization and logging
echo "Installing visualization and logging tools..."
pip install matplotlib tensorboard wandb || print_error "Failed to install visualization tools"

# Computer vision libraries
echo "Installing computer vision libraries..."
pip install kornia opencv-python || print_error "Failed to install CV libraries"

# Try to install classy_vision (may fail, not critical)
echo "Installing classy_vision (optional)..."
if pip install classy_vision 2>/dev/null; then
    echo "  ✓ classy_vision installed"
else
    echo "  ✗ classy_vision failed (not critical, skipping)"
fi

# Additional useful packages
echo "Installing additional packages..."
pip install scipy scikit-learn tqdm || print_error "Failed to install additional packages"

print_success "Additional dependencies installed"

# ============ STEP 10: Clone Scale-MAE repository ============
print_step 10 $TOTAL_STEPS "Cloning Scale-MAE repository"
if [ -d "$REPO_DIR" ]; then
    echo "scale-mae directory already exists. Removing..."
    rm -rf "$REPO_DIR"
fi

git clone https://github.com/bair-climate-initiative/scale-mae.git "$REPO_DIR" || print_error "Failed to clone repository"
cd "$REPO_DIR" || print_error "Failed to enter scale-mae directory"
print_success "Repository cloned to $REPO_DIR"

# ============ STEP 11: Install Scale-MAE package ============
print_step 11 $TOTAL_STEPS "Installing Scale-MAE package"
pip install -e . || print_error "Failed to install Scale-MAE package"

# Verify installation
python << 'EOF' || print_error "Scale-MAE package verification failed"
from mae.models_mae import mae_vit_large_patch16
from mae.models_vit import vit_large_patch16
model = mae_vit_large_patch16()
num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Scale-MAE package installed. Model has {num_params:.1f}M parameters")
EOF

print_success "Scale-MAE package installed"

# ============ STEP 12: Download pretrained weights ============
print_step 12 $TOTAL_STEPS "Downloading pretrained weights"
if [ -f "scalemae-vitlarge-800.pth" ]; then
    echo "Checkpoint already exists. Skipping download."
else
    echo "Downloading scalemae-vitlarge-800.pth (1.29 GB)..."
    wget https://huggingface.co/earthflow/GeoFMs/resolve/main/scalemae-vitlarge-800.pth || print_error "Failed to download checkpoint"
fi

# Verify checkpoint
python << 'EOF' || print_error "Checkpoint verification failed"
import torch
checkpoint = torch.load("scalemae-vitlarge-800.pth", map_location="cpu")
if "model" not in checkpoint:
    raise RuntimeError("Checkpoint missing 'model' key")
num_keys = len(checkpoint["model"].keys())
print(f"Checkpoint loaded successfully. Contains {num_keys} keys")
EOF

print_success "Pretrained weights downloaded and verified"

# ============ Installation Complete ============
echo ""
echo -e "${GREEN}======================================"
echo "Installation Complete!"
echo "======================================${NC}"
echo ""
echo "Summary of what was installed:"
echo "  - Virtual environment: $VENV_DIR"
echo "  - Python packages: PyTorch, timm 0.3.2, GDAL, geospatial libs"
echo "  - Scale-MAE repository: $REPO_DIR"
echo "  - Pretrained weights: $REPO_DIR/scalemae-vitlarge-800.pth"
echo ""
echo "Next steps:"
echo ""
echo "  1. Download EuroSAT dataset:"
echo "     cd $REPO_DIR"
echo "     wget https://zenodo.org/record/7711810/files/EuroSAT.zip"
echo "     unzip EuroSAT.zip"
echo "     ln -s \$(pwd)/EuroSAT data/eurosat"
echo ""
echo "  2. Run pre-flight check:"
echo "     bash preflight_check.sh"
echo ""
echo "  3. Run inference:"
echo "     sbatch inference_job.sh"
echo ""
echo "To reactivate environment later:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $REPO_DIR"
echo ""