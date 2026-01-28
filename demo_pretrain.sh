#!/bin/bash
# Quick demo of pre-training system

echo "🐟 Fish Evolution Pre-training Demo"
echo ""
echo "This will run a quick 50-generation training session"
echo "with visualization of the progress."
echo ""

# Install dependencies if needed
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "Installing numpy..."
    pip3 install numpy
fi

if ! python3 -c "import simpy" 2>/dev/null; then
    echo "Installing simpy..."
    pip3 install simpy
fi

# Run training
echo "Starting training..."
echo ""

python3 pretrain.py \
    -g 50 \
    --fish 80 \
    --foods 3 \
    -o assets/pretrained_weights.json

echo ""
echo "✓ Demo complete!"
echo ""
echo "Your website will now load with pre-trained fish at generation 50+"
echo "Try these commands next:"
echo ""
echo "  # More training (500 generations)"
echo "  python3 pretrain.py -g 500 -o assets/pretrained_weights.json"
echo ""
echo "  # With visualization (requires matplotlib)"
echo "  python3 pretrain.py -v -g 100"
echo ""
echo "  # Custom generation time"
echo "  python3 pretrain.py --gen-time 10.0 -g 100"
echo ""
