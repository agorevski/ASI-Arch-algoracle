#!/usr/bin/env python3
"""
Test script to verify the training pipeline works correctly
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')


def test_deltanet_training():
    """Test that DeltaNet can be loaded and trained"""

    logging.info("🧪 Testing DeltaNet training pipeline...")
    logging.info("=" * 50)

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

        logging.info(f"Running: {' '.join(cmd)}")
        logging.info("-" * 50)

        result = subprocess.run(cmd, capture_output=True, text=True)

        logging.info("STDOUT:")
        logging.info(result.stdout)

        if result.stderr:
            logging.warning("\nSTDERR:")
            logging.warning(result.stderr)

        if result.returncode == 0:
            logging.info("\n✅ Training test completed successfully!")

            # Check if output files were created
            loss_file = Path("files/analysis/loss.csv")
            benchmark_file = Path("files/analysis/benchmark.csv")

            if loss_file.exists():
                logging.info(f"✅ Loss file created: {loss_file}")
                # Show first few lines
                with open(loss_file, 'r') as f:
                    lines = f.readlines()[:5]
                    logging.info("   Sample content:")
                    for line in lines:
                        logging.info(f"   {line.strip()}")
            else:
                logging.warning(f"❌ Loss file not found: {loss_file}")

            if benchmark_file.exists():
                logging.info(f"✅ Benchmark file created: {benchmark_file}")
                # Show content
                with open(benchmark_file, 'r') as f:
                    content = f.read().strip()
                    logging.info("   Content:")
                    for line in content.split('\n'):
                        logging.info(f"   {line}")
            else:
                logging.warning(f"❌ Benchmark file not found: {benchmark_file}")

            return True
        else:
            logging.warning(f"\n❌ Training test failed with return code: {result.returncode}")
            return False

    except Exception as e:
        logging.warning(f"❌ Test failed with exception: {e}")
        return False
    finally:
        # Change back to root directory
        os.chdir("..")


def main():
    """Main test function"""

    logging.info("🔧 ASI-Arch Training Pipeline Test")
    logging.info("=" * 60)

    # Check if required files exist
    required_files = [
        "pipeline/train.py",
        "pipeline/pool/deltanet_base.py",
        "pipeline/config.py"
    ]

    logging.info("\n📋 Checking required files...")
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            logging.info(f"✅ {file_path}")
        else:
            logging.warning(f"❌ {file_path} - NOT FOUND")
            all_files_exist = False

    if not all_files_exist:
        logging.warning("\n❌ Some required files are missing!")
        return False

    # Test training
    logging.info("\n🏃 Running training test...")
    success = test_deltanet_training()

    if success:
        logging.info("\n" + "=" * 60)
        logging.info("🎉 All tests passed! Training pipeline is working correctly.")
        logging.info("   You can now initialize the database and run experiments:")
        logging.info("   1. python init_seed_architecture.py")
        logging.info("   2. cd pipeline && python pipeline.py")
    else:
        logging.warning("\n" + "=" * 60)
        logging.warning("❌ Tests failed. Please check the error messages above.")

    return success


if __name__ == "__main__":
    main()
