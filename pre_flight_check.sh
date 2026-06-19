#!/bin/bash
# Scale-MAE Pre-Flight Verification Script
# This script checks EVERYTHING before you run inference to catch errors early

set -e  # Exit on any error

echo "======================================"
echo "Scale-MAE Pre-Flight Check"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to print success
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
print_error() {
    echo -e "${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

echo "=== 1. Python Environment Check ==="
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ] && [ "$PYTHON_MINOR" -le 10 ]; then
        print_success "Python version: $PYTHON_VERSION (OK)"
    else
        print_error "Python version: $PYTHON_VERSION (Need 3.8-3.10)"
    fi
else
    print_error "Python not found in PATH"
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "No virtual environment activated. Did you run 'source ~/scalemae_env/bin/activate'?"
else
    print_success "Virtual environment activated: $VIRTUAL_ENV"
fi

echo ""
# echo "=== 2. CUDA and GPU Check ==="

# # Check nvidia-smi
# if command -v nvidia-smi &> /dev/null; then
#     print_success "nvidia-smi found"
    
#     # Get GPU info
#     GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
#     GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | awk '{print $1}')
#     CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    
#     echo "  GPU: $GPU_NAME"
#     echo "  Memory: $GPU_MEMORY MB"
#     echo "  CUDA Driver Version: $CUDA_VERSION"
    
#     # Check if memory is sufficient
#     if [ "$GPU_MEMORY" -lt 8000 ]; then
#         print_error "GPU memory ($GPU_MEMORY MB) is insufficient. Need at least 8GB for ViT-Large"
#     else
#         print_success "GPU memory sufficient: $GPU_MEMORY MB"
#     fi
# else
#     print_error "nvidia-smi not found. No GPU available?"
# fi

# Check loaded CUDA module
# if module list 2>&1 | grep -q cuda; then
#     CUDA_MODULE=$(module list 2>&1 | grep cuda | head -1)
#     print_success "CUDA module loaded: $CUDA_MODULE"
# else
#     print_warning "No CUDA module loaded. You may need to run 'module load cuda/11.6'"
# fi

# echo ""
# echo "=== 3. PyTorch Installation Check ==="

# python << 'PYTHON_CHECK_1'
# import sys
# try:
#     import torch
#     print(f"✓ PyTorch version: {torch.__version__}")
    
#     # Check CUDA availability
#     if torch.cuda.is_available():
#         print(f"✓ CUDA available: Yes")
#         print(f"  CUDA version: {torch.version.cuda}")
#         print(f"  GPU count: {torch.cuda.device_count()}")
#         print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        
#         # Test CUDA operations
#         try:
#             x = torch.randn(10, 10).cuda()
#             y = torch.randn(10, 10).cuda()
#             z = torch.matmul(x, y)
#             print("✓ CUDA operations working correctly")
#         except Exception as e:
#             print(f"✗ CUDA operations failed: {e}")
#             sys.exit(1)
#     else:
#         print("✗ CUDA not available in PyTorch")
#         sys.exit(1)
        
# except ImportError as e:
#     print(f"✗ PyTorch not installed: {e}")
#     sys.exit(1)
# PYTHON_CHECK_1

# if [ $? -ne 0 ]; then
#     ERRORS=$((ERRORS + 1))
# fi

# echo ""
# echo "=== 4. Critical Dependencies Check ==="

# python << 'PYTHON_CHECK_2'
# import sys

# # Check timm version
# try:
#     import timm
#     if timm.__version__ == "0.3.2":
#         print(f"✓ timm version: {timm.__version__} (CORRECT)")
#     else:
#         print(f"✗ timm version: {timm.__version__} (Need exactly 0.3.2)")
#         sys.exit(1)
# except ImportError:
#     print("✗ timm not installed")
#     sys.exit(1)

# # Check torchvision
# try:
#     import torchvision
#     print(f"✓ torchvision version: {torchvision.__version__}")
# except ImportError:
#     print("✗ torchvision not installed")
#     sys.exit(1)

# # Check other essentials
# essentials = ['numpy', 'yaml', 'PIL']
# for pkg in essentials:
#     try:
#         __import__(pkg)
#         print(f"✓ {pkg} installed")
#     except ImportError:
#         print(f"✗ {pkg} not installed")
#         sys.exit(1)
# PYTHON_CHECK_2

# if [ $? -ne 0 ]; then
#     ERRORS=$((ERRORS + 1))
# fi

echo ""
echo "=== 5. Geospatial Libraries Check ==="

python << 'PYTHON_CHECK_3'
import sys

# Check GDAL
try:
    from osgeo import gdal
    print(f"✓ GDAL version: {gdal.__version__}")
except ImportError as e:
    print(f"✗ GDAL not installed: {e}")
    print("  Fix: module load gdal && pip install GDAL==$(gdal-config --version)")
    sys.exit(1)

# Check rasterio
try:
    import rasterio
    print(f"✓ rasterio installed")
except ImportError:
    print("✗ rasterio not installed")
    print("  Fix: pip install rasterio")
    sys.exit(1)

# Check shapely
try:
    import shapely
    print(f"✓ shapely installed")
except ImportError:
    print("✗ shapely not installed")
    print("  Fix: pip install shapely")
    sys.exit(1)

# Check geopandas
try:
    import geopandas
    print(f"✓ geopandas installed")
except ImportError:
    print("✗ geopandas not installed")
    print("  Fix: pip install geopandas")
    sys.exit(1)
PYTHON_CHECK_3

if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "=== 6. Scale-MAE Repository Check ==="

# Check if in scale-mae directory
if [ -f "main_pretrain.py" ] && [ -f "main_linprobe.py" ]; then
    print_success "In Scale-MAE repository directory"
else
    print_error "Not in Scale-MAE repository. Run 'cd scale-mae' first"
fi

# Check if Scale-MAE package is installed
python << 'PYTHON_CHECK_4'
import sys
try:
    from mae.models_mae import mae_vit_large_patch16
    from mae.models_vit import vit_large_patch16
    print("✓ Scale-MAE package installed (models import successfully)")
    
    # Try to instantiate model
    try:
        model = mae_vit_large_patch16()
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ Model instantiation successful ({num_params:.1f}M parameters)")
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        sys.exit(1)
        
except ImportError as e:
    print(f"✗ Scale-MAE package not installed: {e}")
    print("  Fix: pip install -e .")
    sys.exit(1)
PYTHON_CHECK_4

if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "=== 7. Pretrained Weights Check ==="

if [ -f "scalemae-vitlarge-800.pth" ]; then
    FILE_SIZE=$(du -h scalemae-vitlarge-800.pth | cut -f1)
    print_success "Checkpoint found: scalemae-vitlarge-800.pth ($FILE_SIZE)"
    
    # Verify checkpoint can be loaded
    python << 'PYTHON_CHECK_5'
import sys
import torch

try:
    checkpoint = torch.load("scalemae-vitlarge-800.pth", map_location="cpu")
    
    if "model" in checkpoint:
        print("✓ Checkpoint has 'model' key")
        num_keys = len(checkpoint["model"].keys())
        print(f"  Number of model keys: {num_keys}")
        
        # Check for essential keys
        essential_keys = ["patch_embed.proj.weight", "blocks.0.attn.qkv.weight", "pos_embed"]
        for key in essential_keys:
            if key in checkpoint["model"]:
                print(f"✓ Found essential key: {key}")
            else:
                print(f"✗ Missing essential key: {key}")
                sys.exit(1)
    else:
        print("✗ Checkpoint missing 'model' key")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Failed to load checkpoint: {e}")
    sys.exit(1)
PYTHON_CHECK_5
    
    if [ $? -ne 0 ]; then
        ERRORS=$((ERRORS + 1))
    fi
else
    print_error "Checkpoint not found: scalemae-vitlarge-800.pth"
    echo "  Download: wget https://huggingface.co/earthflow/GeoFMs/resolve/main/scalemae-vitlarge-800.pth"
fi

echo ""
echo "=== 8. Dataset Check ==="

# Check for data directory
if [ -d "data" ]; then
    print_success "data/ directory exists"
    
    # Check for EuroSAT
    if [ -d "data/eurosat" ]; then
        print_success "EuroSAT dataset found"
        
        # Check structure
        if [ -d "data/eurosat/train" ] && [ -d "data/eurosat/val" ]; then
            print_success "EuroSAT has train/ and val/ directories"
            
            # Count classes
            NUM_CLASSES=$(ls -d data/eurosat/train/*/ 2>/dev/null | wc -l)
            if [ "$NUM_CLASSES" -eq 10 ]; then
                print_success "EuroSAT has 10 classes (correct)"
            else
                print_warning "EuroSAT has $NUM_CLASSES classes (expected 10)"
            fi
        else
            print_error "EuroSAT missing train/ or val/ directories"
        fi
    else
        print_warning "EuroSAT dataset not found at data/eurosat"
        echo "  Download: wget https://zenodo.org/record/7711810/files/EuroSAT.zip && unzip EuroSAT.zip"
    fi
    
    # Check for other datasets
    for dataset in fmow-rgb resisc airound mlrsnet; do
        if [ -d "data/$dataset" ]; then
            print_success "$dataset dataset found"
        fi
    done
else
    print_error "data/ directory not found"
fi

# Check for split files
if [ -d "splits" ]; then
    print_success "splits/ directory exists"
    
    # Check for EuroSAT splits
    if [ -f "splits/eurosat_train.txt" ] && [ -f "splits/eurosat_val.txt" ]; then
        print_success "EuroSAT split files found"
    else
        print_warning "EuroSAT split files not found in splits/"
    fi
else
    print_error "splits/ directory not found"
fi

echo ""
echo "=== 9. Distributed Training Setup Check ==="

# Check torch.distributed
python << 'PYTHON_CHECK_6'
import sys
try:
    import torch.distributed as dist
    print("✓ torch.distributed available")
except ImportError:
    print("✗ torch.distributed not available")
    sys.exit(1)
PYTHON_CHECK_6

if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

# Check if required ports are available
MASTER_PORT=12355
if command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ":$MASTER_PORT "; then
        print_warning "Port $MASTER_PORT is already in use. Change MASTER_PORT in your script"
    else
        print_success "Port $MASTER_PORT is available"
    fi
fi

echo ""
echo "=== 10. Memory and Storage Check ==="

# Check available disk space
AVAILABLE_SPACE=$(df -h . | tail -1 | awk '{print $4}')
print_success "Available disk space: $AVAILABLE_SPACE"

# Check available RAM
if command -v free &> /dev/null; then
    AVAILABLE_RAM=$(free -h | grep "Mem:" | awk '{print $7}')
    print_success "Available RAM: $AVAILABLE_RAM"
fi

echo ""
echo "=== 11. Test Model Loading ==="

python << 'PYTHON_CHECK_7'
import sys
import torch
from mae.models_vit import vit_large_patch16
from mae.pos_embed import interpolate_pos_embed

try:
    print("Loading checkpoint...")
    checkpoint = torch.load("scalemae-vitlarge-800.pth", map_location="cpu")
    
    print("Initializing model...")
    model = vit_large_patch16(img_size=224, num_classes=0, global_pool="avg")
    
    print("Preparing checkpoint...")
    checkpoint_model = checkpoint["model"]
    
    # Remove head keys if present
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model:
            del checkpoint_model[k]
    
    print("Interpolating position embeddings...")
    interpolate_pos_embed(model, checkpoint_model)
    
    print("Loading state dict...")
    model.load_state_dict(checkpoint_model, strict=False)
    
    print("Moving to GPU...")
    model = model.cuda()
    model.eval()
    
    print("Testing forward pass...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
        output = model(dummy_input)
        print(f"✓ Model inference successful! Output shape: {output.shape}")
        
except Exception as e:
    print(f"✗ Model loading/inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_CHECK_7

if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "======================================"
echo "Pre-Flight Check Summary"
echo "======================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED!${NC}"
    echo ""
    echo "You're ready to run inference! Use:"
    echo ""
    echo "  Single GPU:"
    echo "  python -m torch.distributed.launch --nproc_per_node=1 \\"
    echo "      main_pretrain.py --resume scalemae-vitlarge-800.pth --eval_only \\"
    echo "      --eval_dataset eurosat --eval_train_fnames splits/eurosat_train.txt \\"
    echo "      --eval_val_fnames splits/eurosat_val.txt"
    echo ""
    echo "  Or submit SLURM job:"
    echo "  sbatch inference_job.sh"
    echo ""
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ CHECKS PASSED WITH $WARNINGS WARNING(S)${NC}"
    echo "You can proceed, but review warnings above."
elif [ $ERRORS -eq 1 ]; then
    echo -e "${RED}✗ $ERRORS ERROR FOUND${NC}"
    echo "Fix the error above before running inference."
    exit 1
else
    echo -e "${RED}✗ $ERRORS ERRORS FOUND${NC}"
    echo "Fix the errors above before running inference."
    exit 1
fi