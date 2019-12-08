import os, random
random.seed(777)


def k_fold(dataset_dir, n_fold, out_folder=""):
    os.chdir(dataset_dir)
    print("\nCurrent Directory: %s" % os.getcwd())

    datafiles = os.listdir("raw-data/ocr-output")
    print("\n%d OCR files contained as input" % len(datafiles))

    # ########## k-fold spliting ##########
    random.shuffle(datafiles)
    k = n_fold

    portion = int(len(datafiles) / k)
    mod = len(datafiles) % k

    for i in range(k):
        start = i * portion

        if i == (k-1): end = portion * (i + 1) + mod
        else: end = portion * (i + 1)

        temp = datafiles[start: end]

        # print(len(temp), temp[0])
        if not out_folder:
            out_folder = "%d_fold_validation" % n_fold
        try: os.makedirs(out_folder)
        except FileExistsError: pass

        with open(os.path.join(out_folder, "%d_fold.lst") % i, "w") as fp:
            fp.write("\n".join(temp))


invoice_data_dir = "../../Invoice_k_fold"
k_fold(invoice_data_dir, 10)
