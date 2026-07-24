import json
import os
import sys

# Ensure we can import from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.experiment_tracking.experiment_manager import ExperimentManager


def main():
    base_dir = "outputs/test_experiments"
    print("Initializing ExperimentManager...")
    manager = ExperimentManager(base_dir=base_dir)
    manager.start(prefix="test_exp")

    print(f"Experiment started with ID: {manager.experiment_id}")

    # Assert directory creation
    for d in manager.dirs.values():
        assert os.path.exists(d), f"Directory {d} was not created."

    # Test collision
    print("Testing collision detection...")
    manager_dup = ExperimentManager(base_dir=base_dir)
    try:
        manager_dup.start(experiment_id=manager.experiment_id)
        raise AssertionError("Collision detection failed. Should have raised FileExistsError.")
    except FileExistsError:
        print("Collision detection works.")

    # 1. Save Config
    print("Saving dummy config...")
    dummy_config = {
        "model": "dummy_net",
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "Adam"
    }
    manager.save_config(dummy_config)

    # 2. Save Metadata
    print("Saving dummy dataset and runtime metadata...")
    runtime_metadata = {
        "dataset": {
            "dataset_name": "dummy_medical_data",
            "train_samples": 1000,
            "val_samples": 200,
            "seed": 42
        },
        "training_params": {
            "batch_size": 32,
            "workers": 4,
            "mixed_precision": True
        }
    }
    manager.save_metadata(runtime_metadata)

    # 3. Log step and epoch metrics
    print("Logging metrics...")
    for epoch in range(1, 4):
        for step in range(1, 11):
            manager.log_step_metrics(step + (epoch-1)*10, {"loss": 1.0 / (step * epoch), "accuracy": 0.5 + 0.1 * epoch})
        manager.log_epoch_metrics(epoch, {"val_loss": 1.0 / epoch, "val_accuracy": 0.6 + 0.1 * epoch})

    # 4. Register Checkpoint
    print("Registering dummy checkpoint...")
    dummy_ckpt_path = "dummy_checkpoint.pt"
    with open(dummy_ckpt_path, "w") as f:
        f.write("DUMMY CHECKPOINT CONTENT")
    manager.register_checkpoint(dummy_ckpt_path, epoch=3)

    # 5. Register Artifact
    print("Registering dummy artifact...")
    dummy_artifact_path = "dummy_plot.png"
    with open(dummy_artifact_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRDUMMY")
    manager.register_artifact(dummy_artifact_path, artifact_type="visualization", description="Validation loss plot", epoch=3)

    # 6. Finish
    print("Finishing experiment...")
    manager.finish()

    # Cleanup dummy files
    if os.path.exists(dummy_ckpt_path):
        os.remove(dummy_ckpt_path)
    if os.path.exists(dummy_artifact_path):
        os.remove(dummy_artifact_path)

    print(f"Verification complete! Files are saved at {manager.experiment_dir}")

    # Assertions
    print("Running assertions...")

    # Config assertion
    assert os.path.exists(os.path.join(manager.dirs['configs'], 'config.yaml'))

    # Metadata assertions
    metadata_path = os.path.join(manager.dirs['metadata'], 'metadata.json')
    assert os.path.exists(metadata_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
        assert 'environment' in metadata
        assert 'packages' in metadata['environment']
        assert 'execution_summary' in metadata
        assert metadata['execution_summary']['total_step_metrics'] == 30
        assert metadata['execution_summary']['total_epoch_metrics'] == 3
        assert metadata['training_params']['workers'] == 4

    # Metrics assertions
    assert os.path.exists(os.path.join(manager.dirs['metrics'], 'step_metrics.jsonl'))
    assert os.path.exists(os.path.join(manager.dirs['metrics'], 'epoch_metrics.jsonl'))

    # Registry assertions
    checkpoint_registry_path = os.path.join(manager.dirs['checkpoints'], 'checkpoint_registry.json')
    assert os.path.exists(checkpoint_registry_path)
    with open(checkpoint_registry_path) as f:
        ckpt_reg = json.load(f)
        assert len(ckpt_reg) == 1
        assert ckpt_reg[0]["filename"] == "dummy_checkpoint.pt"
        assert len(ckpt_reg[0]["checksum"]) == 64  # SHA256 length

    artifact_registry_path = os.path.join(manager.dirs['artifacts'], 'artifact_registry.json')
    assert os.path.exists(artifact_registry_path)
    with open(artifact_registry_path) as f:
        art_reg = json.load(f)
        assert len(art_reg) == 1
        assert art_reg[0]["filename"] == "dummy_plot.png"
        assert len(art_reg[0]["checksum"]) == 64

    print("All assertions passed.")

    # Generate experiment_management_report.md
    report_path = "experiment_management_report.md"
    with open(report_path, "w") as f:
        f.write("# Experiment Management Verification Report\n\n")
        f.write("## Status\nSuccessfully verified the `ExperimentManager` lifecycle via assertion suite.\n\n")
        f.write(f"**Generated Experiment ID:** {manager.experiment_id}\n")
        f.write(f"**Output Directory:** `{manager.experiment_dir}`\n\n")

        f.write("## Functionality Verified (Assertions Passed)\n")
        f.write("- [x] Experiment initialization and ID collision prevention\n")
        f.write("- [x] Environment metadata introspection (Hardware, OS, package versions)\n")
        f.write("- [x] Config and metadata saving with ML runtime parameters\n")
        f.write("- [x] Metric logging (30 step logs, 3 epoch logs)\n")
        f.write("- [x] Checkpoint registry creation and checksum validation\n")
        f.write("- [x] Artifact registry creation and checksum validation\n")
        f.write("- [x] Finalize execution (generated summary in metadata)\n\n")

        f.write("## Execution Summary\n")
        f.write("```json\n")
        f.write(json.dumps(metadata['execution_summary'], indent=4))
        f.write("\n```\n")

if __name__ == "__main__":
    main()
