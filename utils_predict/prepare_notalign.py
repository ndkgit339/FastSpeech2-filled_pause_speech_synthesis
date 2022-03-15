from pathlib import Path
from tqdm import tqdm
import subprocess

def get_textname_list(text_list_path, textname_list_path):
    with open(text_list_path, "r") as f:
        text_list = [l.strip() for l in f if l.strip()]
    
    textname_list = [t.split(":")[0] for t in text_list if len(t.split(":")[1]) > 0]
    with open(textname_list_path, "w") as f:
        f.write("\n".join(textname_list))

def get_ojtlab(textname_list_path, text_dir, ojtlab_dir, openjtalk_path, dic_path, htsvoice_path, max_process = 5):

    with open(textname_list_path, "r") as f:
        text_names = [l.strip() for l in f]

    text_paths = [text_dir / (name + ".txt") for name in text_names]

    # #OpenJTalkでフルコンテキストラベル生成
    procs = []
    for text_path in tqdm(text_paths):
        ojtlab_path = ojtlab_dir / text_path.name.replace(".txt", ".ojtlab")
        proc = subprocess.Popen([
            openjtalk_path,
            "-x", dic_path,
            "-m", htsvoice_path,
            "-ot", ojtlab_path,
            text_path
        ])
        procs.append(proc)

        if len(procs) % max_process == 0:
            for proc in procs:
                proc.communicate()
            procs.clear()
        
    for proc in procs:
        proc.communicate()

def convert_ojtlab_to_fulllab(ojtlab_dir, fulllab_dir):
    ojtlab_paths = ojtlab_dir.glob("*.ojtlab")

    for ojtlab_path in tqdm(ojtlab_paths):
        with open(ojtlab_path, 'r') as f:
            ojtlab = f.read()

        fulllab = \
            ojtlab.split("[Output label]\n")[1].split("\n[Global parameter]")[0]

        with open(fulllab_dir / (ojtlab_path.stem + ".lab"), "w") as f:
            f.write(fulllab)

def full2monolab(ojtlab_text):
    lines = ojtlab_text.split('[Output label]')[1].split('[Global parameter]')[0].split('\n')
    lines = [x for x in lines if x]
    
    monolab_text = []
    for i, l in enumerate(lines):
        l = l.split(' ')[2]
        l = l.split('-')[1]
        l = l.split('+')[0]
        if l == 'sil':
            if i == 0:
                l = 'silB'
        if l == 'pau':
            l = 'sp'
        monolab_text.append(l+"\n")
    else:
        monolab_text[-1] = 'silE'

    return monolab_text

def convert_lab_available(lab):
    if lab == "sil":
        lab = ""
    elif lab == "A":
        lab = "a"
    elif lab == "I":
        lab = "i"
    elif lab == "U":
        lab = "u"
    elif lab == "E":
        lab = "e"
    elif lab == "O":
        lab = "o"
    elif lab == "cl":
        lab = "q"
    elif lab == "pau":
        lab = "sp"
    elif lab == "v":
        lab = "b"
    return lab

def get_monolab(textname_list_path, ojtlab_dir, jl_in_dir):
    """
    open jtalk で得たフルコンテキストラベルをモノフォンラベルに変換
    """
    with open(textname_list_path, "r") as f:
        text_names = [l.strip() for l in f]

    ojtlab_paths = [ojtlab_dir / (text_name + ".ojtlab") for text_name in text_names]
    for ojtlab_path in tqdm(ojtlab_paths):
        with open(ojtlab_path, "r") as f:
            ojtlab_text = f.read()
            monolab_text = full2monolab(ojtlab_text)
        
        for i in range(len(monolab_text)):
            monolab_text[i] = convert_lab_available(monolab_text[i])

        monolab_path = jl_in_dir / ojtlab_path.with_suffix(".lab").name
        with open(monolab_path, "w") as f:
            f.write("".join(monolab_text))

def process(data_dir, openjtalk_path, dic_path, htsvoice_path, n_jobs=4):

    text_dir = data_dir / 'text'
    ojtlab_dir = data_dir / 'ojtlab'
    jl_in_dir = data_dir / 'jl_in_lab'
    fulllab_dir = data_dir / 'fullcontext_lab'

    text_list_path = data_dir / "utt_filler_list.txt"
    textname_list_path = data_dir / "utt_name.list"

    for d in [text_dir, ojtlab_dir, jl_in_dir, fulllab_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("Get text names...")
    get_textname_list(text_list_path, textname_list_path)
    print("Get full context labels...")
    get_ojtlab(textname_list_path, text_dir, ojtlab_dir, openjtalk_path, dic_path, htsvoice_path, max_process=n_jobs)
    print("Convert ojtlab to full labels...")
    convert_ojtlab_to_fulllab(ojtlab_dir, fulllab_dir)
    print("Convert full to mono labels...")
    get_monolab(textname_list_path, ojtlab_dir, jl_in_dir)

def prepare_notalign(data_dir, n_jobs=8):
    openjtalk_path = '/usr/local/bin/open_jtalk'
    dic_path = '/usr/local/share/open_jtalk/open_jtalk_dic_utf_8-1.11'
    htsvoice_path = '/usr/local/share/hts_voice/hts_voice_nitech_jp_atr503_m001-1.05/nitech_jp_atr503_m001.htsvoice'
    
    data_dir = Path(data_dir)

    print(f"---start data {data_dir.name}---")
    process(data_dir, openjtalk_path, dic_path, htsvoice_path, n_jobs=n_jobs)

if __name__=="__main__":
    prepare_notalign("./")