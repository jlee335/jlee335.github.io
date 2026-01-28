#!/usr/bin/env python3
"""Quick test of parallel training system"""

import sys
sys.path.insert(0, '.')

from pretrain import Config, train_parallel

# Quick test with minimal settings
if __name__ == '__main__':
    print("Testing parallel training with shared gene pool...")
    
    base_config = Config.from_file(
        fish_count=20,  # Small for quick test
        generation_time=5.0  # Short generations
    )
    
    results = train_parallel(
        generations=3,  # Just 3 generations for testing
        base_config=base_config,
        aspect_ratios=[(16, 9), (4, 3)],  # Just 2 aspect ratios
        food_counts=[1, 2],  # Just 2 food counts = 4 total variants
        n_threads=4,
        output_dir="test_output"
    )
    
    print("\n✓ Test complete!")
    print(f"Trained {len(results)} variants")
    for r in results:
        print(f"  {r['variant_name']}: max_fitness={r['max_fitness']:.1f}")
    
    print("\nOutput files:")
    print("  test_output/pretrained_weights.json")
    print("  test_output/training_summary.json")
