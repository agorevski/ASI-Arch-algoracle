#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly
"""

import subprocess
import sys
import os
from pathlib import Path


def test_deltanet_training():
    """Test that DeltaNet can be loaded and trained"""

    print("🧪 Testing DeltaNet training pipeline...")
    print("=" * 50)

    # Change to pipeline directory
    os.chdir("pipeline")

    try:
        # Run training with minimal steps for testing
        cmd = [
            sys.executable, "train.py",
            "--model_file", "pool/deltanet/deltanet_base.py",
            "--num_steps", "10",    # Just 10 steps for testing
            "--batch_size", "4",    # Small batch size
            "--vocab_size", "100",  # Small vocab for speed
            "--seq_len", "32",      # Short sequences
            "--model_name", "DeltaNet-Test"
        ]

        print(f"Running: {' '.join(cmd)}")
        print("-" * 50)

        result = subprocess.run(cmd, capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("\n✅ Training test completed successfully!")

            # Check if output files were created
            loss_file = Path("files/analysis/loss.csv")
            benchmark_file = Path("files/analysis/benchmark.csv")

            if loss_file.exists():
                print(f"✅ Loss file created: {loss_file}")
                # Show first few lines
                with open(loss_file, 'r') as f:
                    lines = f.readlines()[:5]
                    print("   Sample content:")
                    for line in lines:
                        print(f"   {line.strip()}")
            else:
                print(f"❌ Loss file not found: {loss_file}")

            if benchmark_file.exists():
                print(f"✅ Benchmark file created: {benchmark_file}")
                # Show content
                with open(benchmark_file, 'r') as f:
                    content = f.read().strip()
                    print("   Content:")
                    for line in content.split('\n'):
                        print(f"   {line}")
            else:
                print(f"❌ Benchmark file not found: {benchmark_file}")

            return True
        else:
            print(f"\n❌ Training test failed with return code: {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    finally:
        # Change back to root directory
        os.chdir("..")


def main():
    """Main test function"""

    print("🔧 ASI-Arch Training Pipeline Test")
    print("=" * 60)

    # Check if required files exist
    required_files = [
        "pipeline/train.py",
        "pipeline/pool/deltanet_base.py",
        "pipeline/config.py"
    ]

    print("\n📋 Checking required files...")
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_files_exist = False

    if not all_files_exist:
        print("\n❌ Some required files are missing!")
        return False

    # Test training
    print("\n🏃 Running training test...")
    success = test_deltanet_training()

    if success:
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Training pipeline is working correctly.")
        print("   You can now initialize the database and run experiments:")
        print("   1. python init_seed_architecture.py")
        print("   2. cd pipeline && python pipeline.py")
    else:
        print("\n" + "=" * 60)
        print("❌ Tests failed. Please check the error messages above.")

    return success


if __name__ == "__main__":
    main()
