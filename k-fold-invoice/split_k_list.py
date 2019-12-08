import os, random
random.seed(777)


def merge_list(list_of_list):
    out = []
    for l in list_of_list:
        out += l
    return out


def create_train_val(n_folds, out_folder):
    n = len(n_folds)
    for i in range(n):
        fold_i_folder = os.path.join(out_folder, "fold_%d_as_val" % i)

        val = n_folds[i]
        train = merge_list(n_folds[:i] + n_folds[i+1:])

        try: os.makedirs(fold_i_folder)
        except FileExistsError: pass

        with open(os.path.join(fold_i_folder, "train.lst"), "w") as fp:
            fp.write("\n".join(train))

        with open(os.path.join(fold_i_folder, "val.lst"), "w") as fp:
            fp.write("\n".join(val))

        print(fold_i_folder, len(train), len(val))


def k_fold(dataset_dir, n_fold, out_folder=""):
    os.chdir(dataset_dir)
    print("\nCurrent Directory: %s" % os.getcwd())

    # datafiles = os.listdir("raw-data/ocr-output")
    datafiles = os.listdir("generated/all/ocr-labels")
    print("\n%d OCR files contained as input" % len(datafiles))

    # ########## k-fold spliting ##########
    random.shuffle(datafiles)
    k = n_fold

    portion = int(len(datafiles) / k)
    mod = len(datafiles) % k
    all_folds = []

    for i in range(k):
        start = i * portion

        if i == (k-1): end = portion * (i + 1) + mod
        else: end = portion * (i + 1)

        temp = datafiles[start: end]
        all_folds.append(temp)

        # Save list.
        if not out_folder:
            out_folder = "%d_fold_validation" % n_fold
        try: os.makedirs(out_folder)
        except FileExistsError: pass

        with open(os.path.join(out_folder, "%d_fold.lst") % i, "w") as fp:
            fp.write("\n".join(temp))

    create_train_val(all_folds, out_folder)
    return all_folds


invoice_data_dir = "../../Invoice_k_fold"
k_fold(invoice_data_dir, 10)
