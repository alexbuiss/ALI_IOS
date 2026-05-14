#!/usr/bin/env python3
"""
Script to run ALIDDM training for both lower and upper jaws sequentially.
Executes lower jaw first, then upper jaw.
"""

import subprocess
import sys
import time

def run_training(jaw_type):
    """
    Run training for specified jaw type.
    
    Args:
        jaw_type: 'L' for lower jaw, 'U' for upper jaw
    """
    jaw_name = "Lower" if jaw_type == 'L' else "Upper"
    print("\n" + "="*80)
    print(f"                    STARTING {jaw_name.upper()} JAW TRAINING")
    print("="*80 + "\n")
    
    # Build the command
    cmd = [
        'python', 'py/main.py',
        '--jaw', jaw_type
    ]
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {jaw_name} jaw training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {jaw_name} jaw training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {jaw_name} jaw training interrupted by user")
        return False

def main():
    """Main orchestration function"""
    print("\n" + "="*80)
    print("         ALIDDM DUAL-JAW TRAINING ORCHESTRATOR")
    print("="*80)
    print("\nThis script will run training for both jaws sequentially:")
    print("  1. Lower Jaw (L)")
    print("  2. Upper Jaw (U)")
    print("\n" + "="*80 + "\n")
    
    start_time = time.time()
    
    # Run lower jaw training
    print("[1/2] Running Lower Jaw Training...")
    lower_success = run_training('L')
    
    if not lower_success:
        print("\n❌ Lower jaw training failed. Aborting upper jaw training.")
        sys.exit(1)
    
    # Wait a bit between trainings
    print("\n⏳ Waiting 5 seconds before starting upper jaw training...\n")
    time.sleep(5)
    
    # Run upper jaw training
    print("[2/2] Running Upper Jaw Training...")
    upper_success = run_training('U')
    
    if not upper_success:
        print("\n❌ Upper jaw training failed.")
        sys.exit(1)
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("                    DUAL-JAW TRAINING COMPLETE")
    print("="*80)
    print(f"\n✅ Both jaw trainings completed successfully!")
    print(f"   Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
