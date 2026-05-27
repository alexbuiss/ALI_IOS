#!/usr/bin/env python3
"""
Script to run ALIDDM training for both lower and upper jaws with all landmark types.
Executes 4 training runs:
  1. Lower Jaw + Cervical landmarks
  2. Lower Jaw + Occlusal landmarks
  3. Upper Jaw + Cervical landmarks
  4. Upper Jaw + Occlusal landmarks
"""

import subprocess
import sys
import time

def run_training(jaw_type, lm_type):
    """
    Run training for specified jaw and landmark type.
    
    Args:
        jaw_type: 'L' for lower jaw, 'U' for upper jaw
        lm_type: 'C' for cervical landmarks, 'O' for occlusal landmarks
    """
    jaw_name = "Lower" if jaw_type == 'L' else "Upper"
    lm_name = "Cervical" if lm_type == 'C' else "Occlusal"
    print("\n" + "="*80)
    print(f"        STARTING {jaw_name.upper()} JAW + {lm_name.upper()} LANDMARKS TRAINING")
    print("="*80 + "\n")
    
    # Determine the models directory based on jaw type
    models_dir = f'/home/luciacev/Desktop/training ios files/all data/models/{jaw_name}/{lm_name}'
    
    # Build the command
    cmd = [
        'python', 'py/main.py',
        '--jaw', jaw_type,
        '--lm_typ', lm_type,
        '--dir_models', models_dir
    ]
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {jaw_name} jaw + {lm_name} landmarks training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {jaw_name} jaw + {lm_name} landmarks training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {jaw_name} jaw + {lm_name} landmarks training interrupted by user")
        return False

def main():
    """Main orchestration function"""
    print("\n" + "="*80)
    print("    ALIDDM QUAD-RUN TRAINING ORCHESTRATOR (Lower+Upper x Cervical+Occlusal)")
    print("="*80)
    print("\nThis script will run training 4 times sequentially:")
    print("  1. Lower Jaw + Cervical Landmarks (L, C)")
    print("  2. Lower Jaw + Occlusal Landmarks (L, O)")
    print("  3. Upper Jaw + Cervical Landmarks (U, C)")
    print("  4. Upper Jaw + Occlusal Landmarks (U, O)")
    print("\n" + "="*80 + "\n")
    
    start_time = time.time()
    
    # Define all training configurations
    configs = [
        ('L', 'C', "Lower Jaw + Cervical"),
        ('L', 'O', "Lower Jaw + Occlusal"),
        ('U', 'C', "Upper Jaw + Cervical"),
        ('U', 'O', "Upper Jaw + Occlusal"),
    ]
    
    results = {}
    
    for idx, (jaw, lm, description) in enumerate(configs, 1):
        print(f"[{idx}/4] Running {description} Training...")
        success = run_training(jaw, lm)
        results[description] = success
        
        if not success:
            print(f"\n❌ {description} training failed. Aborting remaining trainings.")
            sys.exit(1)
        
        # Wait between trainings (except after the last one)
        if idx < len(configs):
            print(f"\n⏳ Waiting 5 seconds before next training...\n")
            time.sleep(5)
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("                    QUAD-RUN TRAINING COMPLETE")
    print("="*80)
    print(f"\n✅ All 4 training runs completed successfully!")
    print("\nTraining Summary:")
    for desc, success in results.items():
        status = "✅ COMPLETED" if success else "❌ FAILED"
        print(f"  {desc}: {status}")
    print(f"\nTotal time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
