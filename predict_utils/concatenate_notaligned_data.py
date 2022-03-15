from pathlib import Path
from tqdm import tqdm

def concatenate_text(utt_filler_list, breath_para_ids, breath_para_list_path, breath_para_text_dir):
        
    breath_para_text_list = []
    for breath_para_id in tqdm(breath_para_ids):
        breath_para_text = "".join([utt.split(":")[1].replace("(F)", "") for utt in utt_filler_list if utt.startswith(f"{breath_para_id}-")])
        breath_para_text_list.append(f"{breath_para_id}:{breath_para_text}")

        with open(breath_para_text_dir / (breath_para_id + ".lab"), "w") as f:
            f.write(breath_para_text)

    with open(breath_para_list_path, "w") as f:
        f.write("\n".join(breath_para_text_list))

def concatenate_lab(utt_filler_list, breath_para_ids, lab_dir, lab_out_dir, ftag_out_dir):

    for breath_para_id in tqdm(breath_para_ids):
        breath_para_lab_paths = [
            lab_dir / (utt.split(":")[0] + ".lab") 
            for utt in utt_filler_list if utt.split(":")[0].startswith(f"{breath_para_id}-")
        ]

        filler_tags_each_utt = [
            int(utt.split(":")[1].startswith("(F)")) 
            for utt in utt_filler_list if utt.split(":")[0].startswith(f"{breath_para_id}-")
        ]

        new_labs = []
        new_filler_tags_each_phone = []
        for breath_para_lab_path, filler_tag in zip(breath_para_lab_paths, filler_tags_each_utt):
            with open(breath_para_lab_path, "r") as f:
                labs = [l.strip() for l in f if len(l.strip()) > 0]
            
            if len(labs) == 0:
                continue
            
            for i, lab in enumerate(labs):
                # 始端・終端のsilをスキップ
                if i == 0 or i+1 == len(labs):
                    continue
                
                new_labs.append(lab)
                new_filler_tags_each_phone.append(str(filler_tag))
            
        
        with open(lab_out_dir / (breath_para_id + ".lab"), "w") as f:
            f.write("\n".join(new_labs))

        with open(ftag_out_dir / (breath_para_id + ".ftag"), "w") as f:
            f.write(" ".join(new_filler_tags_each_phone))

def concatenate_accent(breath_para_ids, accent_dir, accent_out_dir):

    for breath_para_id in tqdm(breath_para_ids):
        accent_paths = accent_dir.glob(f"{breath_para_id}-*.accent")

        accents = ""
        for accent_path in accent_paths:
            with open(accent_path, 'r') as f:
                accent = f.read()
                accents += accent
        
        with open(accent_out_dir / (breath_para_id + ".accent"), "w") as f:
            f.write(accents)

def concatenate_notaligned_data(data_dir):
    data_dir = Path(data_dir)
    utt_filler_list_path = data_dir / "utt_filler_list.txt"
    breath_para_list_path = data_dir / "breath_para_list.txt"
    breath_para_text_dir = data_dir / "text_breath_para"
    lab_dir = data_dir / "jl_in_lab"
    lab_out_dir = data_dir / "lab_breath_para"
    ftag_out_dir = data_dir / "filler_tag"
    accent_dir = data_dir / "accent"
    accent_out_dir = data_dir / "accent_breath_para"

    for d in [breath_para_text_dir, lab_out_dir, ftag_out_dir, accent_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(utt_filler_list_path, "r") as f:
        utt_filler_list = [utt.strip() for utt in f if len(utt.strip()) > 0]

    breath_para_ids = ["-".join(utt.split("-")[:-1]) for utt in utt_filler_list]
    breath_para_ids = sorted(set(breath_para_ids), key=breath_para_ids.index)

    print("Concatenate text...")
    concatenate_text(utt_filler_list, breath_para_ids, breath_para_list_path, breath_para_text_dir)
    print("Concatenate lab...")
    concatenate_lab(utt_filler_list, breath_para_ids, lab_dir, lab_out_dir, ftag_out_dir)
    print("Concatenate accent...")
    concatenate_accent(breath_para_ids, accent_dir, accent_out_dir)

if __name__=="__main__":

    data_dir = Path("./lecture_data-dereverbed-voice_denoised/speaker2_yoshimi/utokyo_lecture_6")
    utt_filler_list_path = data_dir / "utt_filler_list.txt"
    lab_dir = data_dir / "jl_in_lab"
    lab_out_dir = data_dir / "jl_in_lab_breath_para"
    ftag_out_dir = data_dir / "filler_tag"

    for d in [lab_out_dir, ftag_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(utt_filler_list_path, "r") as f:
        utt_filler_list = [utt.strip() for utt in f if len(utt.strip()) > 0]
    utt_filler_list = sorted(
        utt_filler_list, 
        key=lambda x: (x.split("-")[0], int(x.split("-")[1]), int(x.split("-")[2]), int(x.split(":")[0].split("-")[3]))
    )

    breath_para_ids = ["-".join(utt.split("-")[:3]) for utt in utt_filler_list]
    breath_para_ids = sorted(set(breath_para_ids), key=breath_para_ids.index)

    concatenate_lab(utt_filler_list, breath_para_ids, lab_dir, lab_out_dir, ftag_out_dir)