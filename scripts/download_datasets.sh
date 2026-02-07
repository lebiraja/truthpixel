#!/bin/bash

# AuthentiScan Dataset Download Script
# Downloads 3 datasets: GenImage, CIFAKE, and 140K Faces
# Total size: ~18GB

set -e  # Exit on error

echo "=========================================================================="
echo "  AuthentiScan Dataset Download Script"
echo "=========================================================================="
echo "Total size: ~18GB (GenImage 8GB + CIFAKE 3GB + Faces 7GB)"
echo ""

# Setup
BASE_DIR="data"
DOWNLOADS_DIR="${BASE_DIR}/downloads"

mkdir -p "${DOWNLOADS_DIR}"/{genimage,cifake,faces}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Install tools
echo "=========================================================================="
echo "Installing required tools..."
echo "=========================================================================="

pip install -q kaggle gdown

# Setup Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}ERROR: Kaggle credentials not found!${NC}"
    echo ""
    echo "Please setup Kaggle API:"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Move kaggle.json to ~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

chmod 600 ~/.kaggle/kaggle.json
echo -e "${GREEN}✓ Kaggle credentials found${NC}"

# Download CIFAKE
echo ""
echo "=========================================================================="
echo "Downloading CIFAKE (~3GB, 120K images)"
echo "=========================================================================="

if [ -d "${DOWNLOADS_DIR}/cifake/train" ]; then
    echo -e "${YELLOW}CIFAKE already downloaded, skipping...${NC}"
else
    cd "${DOWNLOADS_DIR}/cifake"
    kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
    unzip -q cifake-real-and-ai-generated-synthetic-images.zip
    rm cifake-real-and-ai-generated-synthetic-images.zip
    cd ../../..
    echo -e "${GREEN}✓ CIFAKE downloaded${NC}"
fi

# Download 140K Faces
echo ""
echo "=========================================================================="
echo "Downloading 140K Real and Fake Faces (~7GB)"
echo "=========================================================================="

if [ -d "${DOWNLOADS_DIR}/faces/real" ] || [ -d "${DOWNLOADS_DIR}/faces/fake" ]; then
    echo -e "${YELLOW}Faces already downloaded, skipping...${NC}"
else
    cd "${DOWNLOADS_DIR}/faces"
    kaggle datasets download -d xhlulu/140k-real-and-fake-faces
    unzip -q 140k-real-and-fake-faces.zip
    rm 140k-real-and-fake-faces.zip
    cd ../../..
    echo -e "${GREEN}✓ 140K Faces downloaded${NC}"
fi

# GenImage semi-automated download with gdown
echo ""
echo "=========================================================================="
echo "Downloading GenImage (~8GB, 400K images, 8 generators)"
echo "=========================================================================="

if [ -d "${DOWNLOADS_DIR}/genimage/real" ] && [ -d "${DOWNLOADS_DIR}/genimage/fake" ]; then
    echo -e "${YELLOW}GenImage already downloaded, skipping...${NC}"
else
    cd "${DOWNLOADS_DIR}/genimage"

    echo "Attempting automated download with gdown..."
    FOLDER_ID="1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"

    # Try gdown
    if gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" --remaining-ok 2>/dev/null; then
        echo -e "${GREEN}✓ GenImage downloaded successfully via gdown!${NC}"
    else
        echo -e "${YELLOW}⚠  Automated download failed. Manual download required.${NC}"
        echo ""
        echo "Manual download instructions:"
        echo "  1. Visit: https://drive.google.com/drive/folders/${FOLDER_ID}"
        echo "  2. Download all subfolders (real/ and fake/ with generators)"
        echo "  3. Extract to: ${DOWNLOADS_DIR}/genimage/"
        echo "  4. Final structure should be:"
        echo "     ${DOWNLOADS_DIR}/genimage/"
        echo "       ├── real/"
        echo "       └── fake/"
        echo "           ├── stable_diffusion_v1.4/"
        echo "           ├── stable_diffusion_v1.5/"
        echo "           ├── midjourney/"
        echo "           ├── glide/"
        echo "           ├── adm/"
        echo "           ├── vqdm/"
        echo "           ├── biggan/"
        echo "           └── wukong/"
        echo ""
        echo "After manual download, re-run this script or proceed to organization."
        cd ../../..
        exit 1
    fi

    cd ../../..
fi

echo ""
echo "=========================================================================="
echo "  Download Summary"
echo "=========================================================================="
echo -e "${GREEN}✓ All downloads complete!${NC}"
echo ""
echo "Downloaded datasets:"
echo "  - CIFAKE: ~3GB"
echo "  - 140K Faces: ~7GB"
echo "  - GenImage: ~8GB"
echo "  Total: ~18GB"
echo ""
echo "=========================================================================="
echo "  Next Steps"
echo "=========================================================================="
echo ""
echo "1. Organize datasets into train/val/test splits:"
echo "   python src/organize_datasets.py --datasets genimage cifake faces"
echo ""
echo "2. Verify organization:"
echo "   python scripts/verify_datasets.py"
echo ""
echo "3. Start training:"
echo "   bash scripts/train_all.sh"
echo ""
echo "=========================================================================="
