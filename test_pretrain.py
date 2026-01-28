#!/usr/bin/env python3
"""
Quick test to verify pretrain.py works correctly
"""

import sys
import json

def test_imports():
    """Test that required modules can be imported"""
    print("Testing imports...", end=" ")
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError:
        print("✗ numpy not found - install with: pip3 install numpy")
        return False
    
    try:
        import simpy
        print("  ✓ simpy")
    except ImportError:
        print("✗ simpy not found - install with: pip3 install simpy")
        return False
    
    try:
        import matplotlib
        print("  ✓ matplotlib (optional, for -v visualization)")
    except ImportError:
        print("  ⚠ matplotlib not found (optional) - install with: pip3 install matplotlib")
    
    return True

def test_quick_simulation():
    """Run a very quick simulation to test functionality"""
    print("\nRunning quick 5-generation test...", end=" ")
    
    # Import after checking numpy is available
    from pretrain import Config, Simulation, NeuralNet
    
    config = Config()
    config.fish_count = 10
    config.food_count = 2
    config.generation_time = 0.1  # Very short
    
    sim = Simulation(config)
    
    # Run 5 generations using SimPy
    for gen in range(5):
        sim.run_generation()
        sim.evolve()
    
    print(f"✓ {sim.generation} generations complete")
    return sim

def test_weight_saving(sim):
    """Test that weights can be saved and loaded"""
    print("Testing weight save/load...", end=" ")
    
    from pretrain import save_weights, NeuralNet, Config
    
    # Save
    test_file = "test_weights.json"
    save_weights(sim, test_file, n_brains=5)
    
    # Load and verify
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    assert 'generation' in data
    assert 'brains' in data
    assert len(data['brains']) == 5
    assert 'W1' in data['brains'][0]
    assert 'W2' in data['brains'][0]
    
    # Try loading a brain
    config = Config()
    brain = NeuralNet.from_dict(config, data['brains'][0])
    
    print("✓")
    
    # Cleanup
    import os
    os.remove(test_file)
    
    return True

def main():
    print("=" * 60)
    print("Fish Evolution Pre-training System - Quick Test")
    print("=" * 60)
    print()
    
    # Test imports
    if not test_imports():
        print("\n✗ Tests failed - install dependencies first")
        sys.exit(1)
    
    # Test simulation
    try:
        sim = test_quick_simulation()
    except Exception as e:
        print(f"\n✗ Simulation test failed: {e}")
        sys.exit(1)
    
    # Test weight saving
    try:
        test_weight_saving(sim)
    except Exception as e:
        print(f"\n✗ Weight save/load test failed: {e}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print()
    print("Ready to use! Try:")
    print("  python3 pretrain.py -g 50 -o assets/pretrained_weights.json")
    print()

if __name__ == '__main__':
    main()
