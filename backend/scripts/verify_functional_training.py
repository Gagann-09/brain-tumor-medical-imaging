import glob
import os
import subprocess
from pathlib import Path


def run_cmd(cmd, env=None):
    print(f"Running: {' '.join(cmd)}")
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)

    # We must set PYTHONPATH so the backend directory is in the path
    env_vars["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

    result = subprocess.run(cmd, env=env_vars, capture_output=True, text=True)  # noqa: S603
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    return True


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoints:
        return None
    # Sort by creation time
    return max(checkpoints, key=os.path.getctime)


def verify_pipeline(script_path: str, args: list[str], ckpt_dir: str, experiment_name: str) -> bool:
    print(f"\n--- Verifying {experiment_name} ---")

    # 1. Run for 2 epochs
    print("Phase 1: Initial Training (2 epochs)")
    cmd1 = ["python", script_path, *args, "--epochs", "2"]
    if not run_cmd(cmd1):
        return False

    # Check that checkpoints exist
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not ckpts:
        print(f"Error: No checkpoints saved in {ckpt_dir}")
        return False
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest_ckpt}")

    # 2. Resume for 1 more epoch
    # We specify --epochs 3 so that it runs from epoch 3 to 3.
    print("Phase 2: Resuming Training (1 epoch)")
    cmd2 = ["python", script_path, *args, "--epochs", "3", "--resume_from", latest_ckpt]
    if not run_cmd(cmd2):
        return False

    print(f"Success: {experiment_name} pipeline verified.")
    return True


def main():
    backend_dir = Path(__file__).resolve().parent.parent
    os.chdir(backend_dir.parent) # Change to project root

    # Ensure dataset is there
    dummy_data = "dummy_brats"
    if not os.path.exists(dummy_data):
        print(f"Dummy dataset not found at {dummy_data}. Please run create_minimal_brats.py first.")
        return

    success = True

    # Test U-Net
    unet_script = "backend/scripts/train_unet_baseline.py"
    unet_args = ["--profile", "development"]
    unet_ckpt_dir = "outputs/unet_baseline_development/checkpoints"
    success &= verify_pipeline(unet_script, unet_args, unet_ckpt_dir, "U-Net Baseline")

    # Test ARMT-GAN
    armt_script = "backend/ai/pipeline/train_armt_gan.py"
    armt_args = ["--data_dir", "dummy_brats"]
    armt_ckpt_dir = "outputs/brats_armt_gan/checkpoints"
    success &= verify_pipeline(armt_script, armt_args, armt_ckpt_dir, "ARMT-GAN")

    if success:
        print("\nAll functional verification steps completed successfully!")

        # Verify ExperimentManager logged the right stuff
        exp_dir = "outputs/experiments"
        if os.path.exists(exp_dir):
            runs = os.listdir(exp_dir)
            print(f"\nExperimentManager runs captured: {len(runs)}")
            for run in runs:
                print(f" - {run}")
                metrics_dir = os.path.join(exp_dir, run, "metrics")
                if os.path.exists(metrics_dir):
                    csvs = glob.glob(os.path.join(metrics_dir, "*.csv"))
                    print(f"   * Metrics logged: {[os.path.basename(c) for c in csvs]}")

    else:
        print("\nFunctional verification failed.")

if __name__ == "__main__":
    main()
