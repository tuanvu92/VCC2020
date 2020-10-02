from utils.common_utils import get_list_of_files
from tqdm import tqdm


def get_speaker_list(file_list, pos=-2):
    speaker_label = []
    for fname in file_list:
        sp = fname.split("/")[pos]
        if sp not in speaker_label:
            speaker_label.append(sp)
    speaker_label.sort()
    return speaker_label


def create_vctk_train_test_list(train_utt_per_speaker=100,
                                test_utt_per_speaker=10,
                                eval_utt_per_speaker=10,
                                n_vctk_speaker=100):
    vctk_train_file_list = []
    vctk_test_file_list = []
    vctk_eval_file_list = []

    vctk_file_list = get_list_of_files("/home/messier/PycharmProjects/data/VCTK/mel24k/")
    vctk_file_list = [fname for fname in vctk_file_list if fname.find(".npy") != -1]
    vctk_speaker_list = get_speaker_list(vctk_file_list)
    vctk_speaker_list = vctk_speaker_list[:n_vctk_speaker]
    vctk_file_list = [fname for fname in vctk_file_list if fname.split("/")[-2] in vctk_speaker_list]
    vctk_file_list.sort()
    vctk_speaker_n_train_utt = {sp: 0 for sp in vctk_speaker_list}
    vctk_speaker_n_test_utt = {sp: 0 for sp in vctk_speaker_list}
    vctk_speaker_n_eval_utt = {sp: 0 for sp in vctk_speaker_list}

    vctk_speaker_file_list = dict()

    for speaker in vctk_speaker_list:
        vctk_speaker_file_list[speaker] = [fname for fname in vctk_file_list if fname.split("/")[-2] == speaker]
        vctk_train_file_list.extend(vctk_speaker_file_list[speaker][30:])
        vctk_eval_file_list.extend(vctk_speaker_file_list[speaker][20:30])
        vctk_test_file_list.extend(vctk_speaker_file_list[speaker][:20])
    
    print("train list size: ", len(vctk_train_file_list))
    print("eval list size: ", len(vctk_eval_file_list))
    print("test list size: ", len(vctk_test_file_list))
    with open("../file_lists/vctk_file_list.txt", "w") as f:
        for fname in vctk_file_list:
            f.write("%s\n" % fname)
    with open("../file_lists/mel/vctk_train_list.txt", "w") as f:
        for fname in vctk_train_file_list:
            f.write("%s\n" % fname)
    with open("../file_lists/mel/vctk_test_list.txt", "w") as f:
        for fname in vctk_test_file_list:
            f.write("%s\n" % fname)
    with open("../file_lists/mel/vctk_eval_list.txt", "w") as f:
        for fname in vctk_eval_file_list:
            f.write("%s\n" % fname)


def jvs_file_filter(fname):
    if fname.find(".npy") != -1 and \
            fname.find("falset10") == -1 and \
            fname.find("whisper10") == -1:
        return True
    return False


def create_jvs_train_test_list(train_utt_per_speaker=120,
                               test_utt_per_speaker=5,
                               eval_utt_per_speaker=5,
                               n_jvs_speaker=100):
    jvs_train_file_list = []
    jvs_test_file_list = []
    jvs_eval_file_list = []

    jvs_file_list = get_list_of_files("/home/messier/PycharmProjects/data/jvs_ver1/sp/")
    jvs_file_list = [fname for fname in jvs_file_list if jvs_file_filter(fname)]
    jvs_speaker_list = get_speaker_list(jvs_file_list, pos=-4)
    jvs_speaker_list = jvs_speaker_list[:n_jvs_speaker]
    jvs_file_list = [fname for fname in jvs_file_list if fname.split("/")[-4] in jvs_speaker_list]

    jvs_speaker_n_train_utt = {sp: 0 for sp in jvs_speaker_list}
    jvs_speaker_n_test_utt = {sp: 0 for sp in jvs_speaker_list}
    jvs_speaker_n_eval_utt = {sp: 0 for sp in jvs_speaker_list}

    for fname in tqdm(jvs_file_list):
        sp = fname.split("/")[-4]
        if jvs_speaker_n_train_utt[sp] < train_utt_per_speaker:
            jvs_train_file_list.append(fname)
            jvs_speaker_n_train_utt[sp] += 1
        elif jvs_speaker_n_test_utt[sp] < test_utt_per_speaker:
            jvs_test_file_list.append(fname)
            jvs_speaker_n_test_utt[sp] += 1
        elif jvs_speaker_n_eval_utt[sp] < eval_utt_per_speaker:
            jvs_eval_file_list.append(fname)
            jvs_speaker_n_eval_utt[sp] += 1

    print("jvs train and test list exceeded maximum number")
    print("train list size: ", len(jvs_train_file_list))
    print("test list size: ", len(jvs_test_file_list))

    with open("../file_lists/sp/jvs_train_list.txt", "w") as f:
        for fname in jvs_train_file_list:
            f.write("%s\n" % fname)
    with open("../file_lists/sp/jvs_test_list.txt", "w") as f:
        for fname in jvs_test_file_list:
            f.write("%s\n" % fname)
    with open("../file_lists/sp/jvs_eval_list.txt", "w") as f:
        for fname in jvs_eval_file_list:
            f.write("%s\n" % fname)


if __name__ == "__main__":
    create_vctk_train_test_list()
    # create_jvs_train_test_list()
