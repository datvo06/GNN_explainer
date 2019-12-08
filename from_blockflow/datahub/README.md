# Data processing library 

This repo is for document data processing and augmenting.

## INSTALLATION

Currently, we do not have setup mechanism for this repo. It's comming soon.

## DATA PREPARATION

Please download the project data at here: https://cinnamon-lab.app.box.com/folder/86814448343

Then, decompressing them and run the `process-data.py` with your arguments.

## DEMO

1. Standardizing data:

Suppose that you are interested in `Okaya` dataset, run the following command:
```
python process_data.py --name_norm --path data\okaya\ --selected data\okaya\train.lst
```

This command results a directory in the `generated` directory, with corresponding datetime of running the script. The final data directory is in `mixed-qa-ocr-labels` if the you specified the `--not_mix_ocr` argument. Otherwise, the final data is in `qa-labels-fk-mapped`.

If you do not specify the `--corpus` argument, it will automatically generate the corpus based on algorithms. Also, if the `--not_mix_ocr` argument is not selected, it will automatically label the OCR samples and mixed them in the `mixed-qa-ocr-labels`.

You can run the following command for argument description:
```
python process_data.py --h
```

2. Data augmentation

Suppose that you are interested in `Invoice` dataset and have already run the `process_data.py` for this dataset, which result in `generated\20190905-232845-train` directory. You want to augment the QA with mapped Formal key generated dataset `qa-labels-fk-mapped`, run the following command:
```
python augmenting\augment_data.py --path data\invoice\generated\20190905-232845-train\qa-labels-fk-mapped --classes data\invoice\classes.json --corpus data\invoice\generated\20190905-232845-train\corpus-info\corpus.json --selected data\invoice\train.lst
```

You can run the following command for argument description:
```
python augmenting\augment_data.py --h
```

p.s.

There will be 7 subfolders from output:
```
+-- generated
|   +-- datetime_stamp
|      +-- corpus-info
|      +-- corpus-info-qa
|      +-- mixed-qa-ocr-labels------*
|      +-- ocr-labels---------------*
|      +-- ocr-output-standard
|      +-- qa-labels-fk-mapped
|      +-- qa-labels-standard-------*
```

Of which, the three folders with (*):

`mixed-qa-ocr-labels`, `ocr-labels`, `qa-labels-standard`

are for debug uses.

