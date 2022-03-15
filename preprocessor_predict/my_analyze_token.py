# Python
import copy

# 言語処理
from pyknp import Juman

def get_morph(text):
    morphs = []

    juman = Juman()
    result = juman.analysis(text)

    for m in result.mrph_list():
        morphs.append({
            "type": "L", 
            "surface": m.midasi,
            "reading_form": m.yomi,
            "dictionary_form": m.genkei,
            "pos": m.hinsi
        })

    return morphs

def tagtext_to_tagcharacters(tag_text, start_tag):

    tag_stack = start_tag
    characters = []
    tag = False
    i = 0
    i_char = 0
    f_start_pos = []
    while i < len(tag_text):
        c = tag_text[i]

        if c == "(":
            i += 1
            if tag_text[i:i+2] == "D2":
                c = "D2"
                tag_stack.append(c)
                i += 1
            elif tag_text[i] in ["A","K","W","B"]:
                c = tag_text[i]
                tag_stack.append(c+"f")
            else:
                c = tag_text[i]
                tag_stack.append(c)
                if c == "F":
                    f_start_pos.append(i_char+1)

        elif c == ";":
            tag = tag_stack.pop()
            tag_stack.append(tag[0]+"l")

        elif c == ")":
            tag_stack.pop()

        else:
            tag_stack_copy = copy.copy(tag_stack)
            characters.append((c, tag_stack_copy))
            i_char += 1

        i += 1
    
    return characters, tag_stack

def tagcharacters_to_cleantext(characters, remove_tags={"F", "D", "D2", "X", "Al", "Kf", "Wf", "Bf", "L"}):
    
    clean_text = ""
    for c in characters:
        if len(remove_tags & set(c[1])) == 0:
            if "?" in c[1]:
                clean_text += "?"
            else:
                clean_text += c[0]
                
    return clean_text

# def tagcharacters_to_textwithf(characters):
    
#     textwithf = ""
#     for c in characters:
#         if len({"D","D2","X","Al","Kf","Wf","Bf","L"} & set(c[1])) == 0:
#             if "?" in c[1]:
#                 textwithf += "?"
#             else:
#                 textwithf += c[0]
                
#     return textwithf

def characters_to_fpinfo(characters, remove_tags={"D","D2","X","Al","Kf","Wf","Bf","L"}):

    fp_list = []
    fp = ""
    f_tag = False
    for c in characters:
        if "F" in c[1]:
            fp += c[0]
            if not f_tag:
                f_tag = True
        elif f_tag:
            fp_list.append(fp)
            fp = ""
            f_tag = False
    if f_tag:
        fp_list.append(fp)
        fp = ""
        f_tag = False

    f_start_pos = []
    f_tag = False
    i_char = 0
    for c in characters:
        # fpの開始位置ならリストに追加
        if "F" in c[1]:
            if not f_tag:
                f_start_pos.append(i_char)
                f_tag = True
        elif f_tag:
            f_tag = False

        # 何文字目かをカウント
        if len(remove_tags & set(c[1])) == 0:
            i_char += 1
                
    return fp_list, f_start_pos

def cleantext_to_morphwithfp(
    clean_text, fp_list, f_start_pos):

    morph = get_morph(clean_text)
    morph_out = []
    i_char = 0
    i_m = 0
    for f,f_pos in zip(fp_list,f_start_pos):

        while i_char<f_pos:
            morph_out.append(morph[i_m])
            i_char += len(morph[i_m]["surface"])
            i_m += 1

        if i_char>=f_pos:
            s = "intermid" if i_char>f_pos else None
            morph_out.append({
                "type": "F",
                "surface": f,
                "in_bunse_or_not": s
            })
            i_char += len(f)
    
    while i_m<len(morph):
        morph_out.append(morph[i_m])
        i_m += 1

    return morph_out

def tagtext_to_morphwithfp(
    tagtext, start_tag, 
    remove_tags={"D", "D2", "X", "Al", "Kf", "Wf", "Bf", "L"}):

    tagcharacters, end_tag = tagtext_to_tagcharacters(
        tagtext, start_tag)
    cleantext = tagcharacters_to_cleantext(
        tagcharacters, remove_tags=remove_tags|{"F"})
    fp_list, f_start_pos = characters_to_fpinfo(
        tagcharacters, remove_tags=remove_tags)
    
    morphwithfp = cleantext_to_morphwithfp(
        cleantext, fp_list, f_start_pos)
    
    return tagcharacters, cleantext, end_tag, morphwithfp

class IPU:
    def __init__(
        self, 
        ipu_id,
        ipu_textlist_dict, 
        start_tag, 
        remove_tags,
    ):
                
        tag_text_list = ipu_textlist_dict[ipu_id]
        self.tag_text = ''.join(tag_text_list)

        # Rタグ（個人情報，差別用語など）が含まれるかどうか
        self.r_tag = True if "(R" in self.tag_text else False

        # ?タグ（聞き取りに自信がない）が含まれるかどうか
        self.hatena_tag = True if "(R" in self.tag_text else False
        
        # <FV>タグ（聞き取りに自信がない）が含まれるかどうか
        self.fv_tag = True if "<FV>" in self.tag_text else False

        self.tagcharacters, self.clean_text, self.end_tag, self.morph_withf = \
            tagtext_to_morphwithfp(
                self.tag_text, start_tag, remove_tags=remove_tags)

def latter_id(s):
    n = int(s)
    n += 1
    if 0<=n<10:
        return '000' + str(n)    
    elif 10<=n<100:
        return '00' + str(n)    
    elif 100<=n<1000:
        return '0' + str(n)    
    else:
        return str(n)

def get_ipu_dict(trn_text_lines):

    trn_text_lines = [l.replace(' ','').split('&')[0] for l in trn_text_lines]

    ipu_textlist_dict = dict()
    
    i = 0
    ipuid = '0001'
    while i < len(trn_text_lines):
        if trn_text_lines[i][:4] == ipuid:            
            ipu_text = []
            i += 1
            while i<len(trn_text_lines) and trn_text_lines[i][:4]!=latter_id(ipuid):
                ipu_text.append(trn_text_lines[i])
                i += 1
            ipu_textlist_dict[ipuid] = ipu_text
                            
            ipuid = latter_id(ipuid)
        else:
            i += 1

    return ipu_textlist_dict

def get_morpheme_with_fptag(
    speaker_id, 
    koen_id, 
    transcription_path, 
    remove_tags, 
    ):
    """Get IDs and segmented morpheme sequence with fp tags.
    
    Parameters
    ----------
    speaker_id: str
        id of speaker
    koen_id: str
        id of koen
    transcription_path: str
        path to TRN (transcription) file of each koen in CSJ
    remove_tags: set, defualt={"D", "D2", "X", "Al", "Kf", "Wf", "Bf", "L"}
        set of disfluency tags to remove
    tokenizer_name: {"sudachipy", "juman"}, default="juman"

    Returns
    -------
    morphs: list of tuple
        list of tuple (speaker ID, koen ID, IPU ID, morpheme sequence)
    """

    # Get dictionary of {ID of IPU: text of IPU}
    with open(transcription_path, "r", encoding="shift-jis") as f:
        trn_text_lines = f.readlines()        
    trn_text_lines = [l for l in trn_text_lines if l!="" and l[0]!="%"]
    ipu_textlist_dict = get_ipu_dict(trn_text_lines)

    # Get IDs and segmented morpheme sequence with fps
    ipu_list = []
    ipu_id = "0001"
    start_tag = []
    for ipu_id in ipu_textlist_dict.keys():
        # Get information of each IPU
        ipu = IPU(
            ipu_id,
            ipu_textlist_dict,
            start_tag,
            remove_tags,
            )

        # Skip if IPU includes "R" or "?" tag
        if ipu.r_tag or ipu.hatena_tag or ipu.fv_tag:
            continue

        # Get list of morpheme texts in IPU
        ipu_text = []
        for m in ipu.morph_withf:
            if m["type"]=="F":
                ipu_text.append("(F{})".format(m["surface"]))
            else:
                ipu_text.append(m["surface"])

        # Add (speaker ID, koen ID, IPU ID, morpheme sequence) to list of IPU
        ipu_list.append((
            speaker_id, koen_id, ipu_id, " ".join(ipu_text)
        ))
        start_tag = ipu.end_tag
        ipu_id = latter_id(ipu_id)
            
    return ipu_list

if __name__ == '__main__':
    # tag_text = "(Fえー)まずですね(Fえー)このような(Fまー)テーマに(Fえー)私共が取り組ませていただいた(Fまー)バックグラウンドでございますけれども(Fえー)(Fま)大きく三点ございまして(Fま)一点目はですね(Fえ)九十七年頃からですね(Fま)私共は(Fあっ)私共は(Fあのー)専門学校を運営してる学校法人なんですけれども(Fえ)九十七年からですね(Fえー)(Aダブリュービーティー;ＷＢＴ)を(Fまー)取り組んで(Fあ)(Fえー)従来の(Fお)学校教育に取り込めないだろうかと(Fえ)いうことを(Fまー)色々やってまいりました"
    # start_tag = []
    # tag_text = "頼兼公の仁心にて大川にて花火を(Fえ)灯し給う)これは(Fあの)"
    # start_tag = ["O"]
    # tag_text = "そうですねびっくりしましたけどねそうかそれはでもいい経験(Fあ)そうそう(Fうん)ですね"
    # start_tag = []
    # tag_text = "(Fえー)ただ今(Fあの)御紹介いただきました(Fえー)電子学園の(R×××)と申します"
    # start_tag = []
    
    # print(tag_text)
    # _, cleantext, _, m, b = tagtext_to_tokenwithfp(tag_text,start_tag)
    # for im in m:
    #     print(im)
    #print(len([im[1] for im in m if im[0]=="F"]))
    # print(b)
    #print(len([ib[1] for ib in b if ib[0]=="F"]))

    pass