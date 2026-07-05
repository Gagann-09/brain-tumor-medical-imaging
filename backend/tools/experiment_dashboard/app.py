import streamlit as st
import json
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="ARMT-GAN Experiment Dashboard", layout="wide")
st.title("Experiment Comparison Dashboard")

outputs_dir = Path("outputs")

if not outputs_dir.exists():
    st.warning("Outputs directory not found. Please run experiments first.")
    st.stop()

experiments = []
for exp_dir in outputs_dir.iterdir():
    if not exp_dir.is_dir():
        continue
        
    prov_path = exp_dir / "provenance.json"
    card_path = exp_dir / "model_card.json"
    
    exp_data = {"Experiment Name": exp_dir.name}
    
    if prov_path.exists():
        with open(prov_path, "r") as f:
            prov = json.load(f)
            
        exp_data["Dataset ID"] = prov.get("dataset", {}).get("registry_id")
        exp_data["Dataset Fingerprint"] = prov.get("dataset", {}).get("fingerprint")
        exp_data["Git Commit"] = prov.get("git", {}).get("commit_sha")
        exp_data["GPU"] = prov.get("hardware", {}).get("gpu")
        exp_data["Seed"] = prov.get("execution", {}).get("seed")
        
    if card_path.exists():
        with open(card_path, "r") as f:
            card = json.load(f)
            
        metrics = card.get("metrics", {})
        exp_data["Dice"] = metrics.get("val_dice_mean", None)
        exp_data["IoU"] = metrics.get("iou_mean", None)
        exp_data["Hausdorff"] = metrics.get("hausdorff_95_mean", None)
        exp_data["ASSD"] = metrics.get("assd_mean", None)
        exp_data["Model Architecture"] = card.get("model_details", {}).get("architecture")
        
    experiments.append(exp_data)

if not experiments:
    st.info("No experiments with provenance/model cards found.")
else:
    df = pd.DataFrame(experiments)
    st.dataframe(df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if "Dice" in df.columns:
            st.bar_chart(df.set_index("Experiment Name")["Dice"])
    with col2:
        if "Hausdorff" in df.columns:
            st.bar_chart(df.set_index("Experiment Name")["Hausdorff"])
