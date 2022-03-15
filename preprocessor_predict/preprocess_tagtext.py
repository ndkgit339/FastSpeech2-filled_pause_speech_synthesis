from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# My library
from .my_analyze_token import get_morpheme_with_fptag

def process_tagtext(config):
    # Read person list file
    with open(Path(config.corpus_dir) / "speaker.list", "r") as f:
        person_ids = [l.strip() for l in f if len(l.strip()) > 0]
        
    # Get list of speaker ids
    with open(Path(config.corpus_dir) / "speaker_koen.list") as f:
        speakers = [
            speaker.strip() for speaker in f 
            if speaker.split(":")[0] in person_ids]

    # Set directory
    trn_dir = Path(config.corpus_dir) / "TRN"
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get speaker ID, koen ID, and transcription file path
    speakerid_koenid_trnpath_list = []
    for speaker in speakers:
        speaker_id = speaker.split(":")[0]
        koen_ids = speaker.split(":")[1].split(",")
        for koen_id in koen_ids:
            # Only "学会講演(A)" or "模擬講演(S)"
            if koen_id[0] in ["A", "S"]:
                speakerid_koenid_trnpath_list.append(
                    (speaker_id, koen_id, list(trn_dir.glob(f"**/{koen_id}.trn"))[0])
                )
    
    # Set of disfluency tags to remove
    remove_tags = {
        "D",    # 言い直し語断片
        "D2",   # 助詞言い直し
        "X",    # 非朗読対象発話
        "Al",   # アルファベット,数字表記を除き，アルファベットの読みを残す
        "Kf",   # 漢字表記出来なかった場合，漢字表記残す
        "Wf",   # 発音エラーを除き，正しい発音残す
        "Bf",   # 知識レベルの言い間違いを除き，正しいのを残す
        "L"     # ささやき声など
    }

    # Multi processing
    with ProcessPoolExecutor(config.n_jobs) as executor:
        futures = [
            executor.submit(
                get_morpheme_with_fptag,
                speaker_id,
                koen_id,
                trn_path,
                remove_tags,
            )
            for speaker_id, koen_id, trn_path in speakerid_koenid_trnpath_list
        ]
   
        ipu_list = []
        for future in tqdm(futures):
            ipu_list += future.result()

        print(f"num of breath groups: {len(ipu_list)}")
        with open(out_dir / "ipu.list", "w") as f:
            f.write("\n".join([":".join(ipu) for ipu in ipu_list]))

@hydra.main(config_path="conf/preprocess", config_name="config")
def myapp(config: DictConfig):
    process_tagtext(config)

if __name__=="__main__":
    myapp()