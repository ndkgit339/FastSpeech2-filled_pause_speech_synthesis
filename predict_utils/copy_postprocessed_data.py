from pathlib import Path
import shutil

def copy_postprocessed_data(src_predicted_dir, target_preprocessed_dir, target_raw_dir, speaker_name):
    
    accent_dir = Path(src_predicted_dir) / "accent_breath_para"
    fptag_dir = Path(src_predicted_dir) / "filler_tag"
    lab_dir = Path(src_predicted_dir) / "lab_breath_para"
    text_dir = Path(src_predicted_dir) / "text_breath_para"

    for p in accent_dir.glob("*.accent"):
        (Path(target_preprocessed_dir) / "accent").mkdir(
            exist_ok=True, parents=True)
        shutil.copy(p, Path(target_preprocessed_dir) / "accent" / p.name)

    for p in fptag_dir.glob("*.ftag"):
        (Path(target_preprocessed_dir) / "fp_tag" / speaker_name).mkdir(
            exist_ok=True, parents=True)
        shutil.copy(
            p, 
            Path(target_preprocessed_dir) / "fp_tag" / speaker_name / p.name)

    for p in lab_dir.glob("*.lab"):
        (Path(target_preprocessed_dir) / "lab" / speaker_name).mkdir(
            exist_ok=True, parents=True)
        shutil.copy(
            p, 
            Path(target_preprocessed_dir) / "lab" / speaker_name / p.name)

    for p in text_dir.glob("*.lab"):
        (Path(target_raw_dir) / speaker_name).mkdir(
            exist_ok=True, parents=True)
        shutil.copy(p, Path(target_raw_dir) / speaker_name / p.name)
