#!/bin/bash
# Demo script for parallel fish evolution training

echo "🐟 Fish Evolution - Parallel Training Demo"
echo "==========================================="
echo ""
echo "This script demonstrates parallel training with:"
echo "  - Multiple aspect ratios (16:9, 4:3, 21:9, 1:1)"
echo "  - Varying food counts (1, 2, 3)"
echo "  - K parallel threads"
echo ""

# Default values
GENERATIONS=50
FISH=100
THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--generations)
            GENERATIONS="$2"
            shift 2
            ;;
        -f|--fish)
            FISH="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -g, --generations N   Number of generations (default: 50)"
            echo "  -f, --fish N         Number of fish (default: 100)"
            echo "  -t, --threads K      Number of parallel threads (default: auto)"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Generations: $GENERATIONS"
echo "  Fish count:  $FISH"
echo "  Threads:     $THREADS"
echo ""
echo "Press Enter to start training..."
read

# Run parallel training
python pretrain.py \
    --parallel \
    --threads $THREADS \
    --generations $GENERATIONS \
    --fish $FISH \
    --aspect-ratios 16:9 4:3 21:9 1:1 \
    --food-range 1 2 3

echo ""
echo "✓ Training complete!"
echo ""
echo "Best global brains saved to: pretrained_weights.json"
echo "Training summary saved to: training_summary.json"
echo ""
echo "Next steps:"
echo "  1. Review training_summary.json"
echo "  2. cp pretrained_weights.json assets/"
echo "  3. Reload your website!"
echo ""
