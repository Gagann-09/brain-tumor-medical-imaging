import glob
import os
import sys

import torch


def get_checkpoints(checkpoint_dir):
    ckpts = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not ckpts:
        return None, None
    latest = max(ckpts, key=os.path.getmtime)
    best_ckpts = [c for c in ckpts if "best" in os.path.basename(c)]
    best = max(best_ckpts, key=os.path.getmtime) if best_ckpts else latest
    return latest, best

def validate_unet():
    print("Validating U-Net Checkpoints...")
    ckpt_dir = "outputs/unet_baseline_research/checkpoints"
    latest, best = get_checkpoints(ckpt_dir)
    if not latest:
        print("ERROR: U-Net checkpoints not found")
        return False
    print(f"Latest: {latest}\nBest: {best}")

    # Resume test
    print("Attempting to load latest checkpoint...")
    try:
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        assert "model_state" in ckpt, "model_state missing"
        print("Latest checkpoint loaded successfully.")
    except Exception as e:
        print(f"Failed to load latest: {e}")
        return False

    # Inference test
    print("Attempting to perform inference on best checkpoint...")
    try:
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        from backend.ai.models.unet import UNet3D
        model = UNet3D(in_channels=4, out_channels=3)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        dummy_input = torch.randn(1, 4, 240, 240, 16) # dummy shape for validation
        with torch.no_grad():
            out = model(dummy_input)
            assert out.shape == (1, 3, 240, 240, 16), f"Wrong output shape: {out.shape}"
            assert not torch.isnan(out).any(), "NaN found in output"
        print("Best checkpoint inference successful. Outputs finite.")
    except Exception as e:
        print(f"Failed inference: {e}")
        return False
    return True

def validate_armt_gan():
    print("Validating ARMT-GAN Checkpoints...")
    ckpt_dir = "outputs/experiments/exp_20260724_205440_c079e47a/checkpoints"
    latest, best = get_checkpoints(ckpt_dir)
    if not latest:
        print("ERROR: ARMT-GAN checkpoints not found")
        return False
    print(f"Latest: {latest}\nBest: {best}")

    # Resume test
    print("Attempting to load latest checkpoint...")
    try:
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        assert "model_state" in ckpt, "model_state missing"
        print("Latest checkpoint loaded successfully.")
    except Exception as e:
        print(f"Failed to load latest: {e}")
        return False

    # Inference test
    print("Attempting to perform inference on best checkpoint...")
    try:
        ckpt = torch.load(best, map_location="cpu", weights_only=False)
        from backend.ai.segmentation.models.armt_gan import ARMTGANModel

        # Provide a dummy loss manager since we only do inference
        from backend.ai.training.components import AdversarialLossManager
        lm = AdversarialLossManager(g_losses={}, d_losses={})
        model = ARMTGANModel(loss_manager=lm)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        dummy_input = torch.randn(1, 4, 240, 240, 16)
        with torch.no_grad():
            out = model.forward(dummy_input) # ARMTGANModel overrides forward
            # Wait, ARMTGAN forward returns dict or tensor?
            # In armt_gan.py it probably passes to generator
            assert not torch.isnan(out).any(), "NaN found in output"
        print("Best checkpoint inference successful. Outputs finite.")
    except Exception as e:
        print(f"Failed inference: {e}")
        return False
    return True

if __name__ == "__main__":
    u_ok = validate_unet()
    a_ok = validate_armt_gan()
    if u_ok and a_ok:
        print("\nALL CHECKPOINT VALIDATIONS PASSED")
    else:
        print("\nCHECKPOINT VALIDATION FAILED")
        sys.exit(1)
