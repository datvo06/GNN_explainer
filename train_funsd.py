from __future__ import print_function, unicode_literals, division
import torch, pickle, time, matplotlib
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset
import os


import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import utils.math_utils as math_utils
import utils.io_utils as io_utils

from utils.pickle_related import read_pickle
from models_funsd import FUNSDModelConfig1
# from models_funsd import FUNSDModelConfig2 as FUNSDModelConfig1

import json
import random
# import shutil
__author__ = "Marc"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FunsdDataLoader(Dataset):
    def __init__(self, funsd_pickle_path, labels_dict=None):
        self.inp_list = pickle.load(open(funsd_pickle_path, 'rb'))
        if labels_dict is None:
            self.labels = dict(
                [label, idx]
                for idx, label in enumerate(
                    list(
                        set(self.inp_list[0]['labels']))))

            json.dump(self.labels, open('labels', 'w'))
        else:
            self.labels = labels_dict
        for each_dict in self.inp_list:
            each_dict['labels'] = np.array(
                [self.labels[label]
                 for label in each_dict['labels']])

    def __len__(self):
        # TODO: features.shape = (N_Graph, Node, Feature)
        return len(self.inp_list)

    def getitem(self, idx):
        return {
            "ocr_values": [cell.ocr_value for cell in self.inp_list[idx]['cells']],
            "adj": torch.Tensor(self.inp_list[idx]['adj_mats']).unsqueeze(0),
            # "feats": torch.Tensor(np.concatenate(
            #   (self.inp_list[idx]['transformer_feature'],
            #    self.inp_list[idx]['pos_feats']), -1)).unsqueeze(0),
            "feats": torch.Tensor(self.inp_list[idx]['transformer_feature']).unsqueeze(0),
            # "feats": torch.Tensor(self.inp_list[idx]['pos_feats']).unsqueeze(0),
            "label": torch.Tensor(self.inp_list[idx]['labels']).unsqueeze(0)
        }

    def __getitem__(self, idx):
        if type(idx) != int:
            list_idx = idx[:]
            return_lists = [self.getitem(idx) for idx in list_idx]
        else:
            return_lists = self.getitem(idx)
        return return_lists


def train(dataset, model_instance, args, same_feat=True,
          val_dataset=None,
          test_dataset=None,
          writer=None,
          mask_nodes=True,
          ):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_instance.parameters()), lr=0.0001
    )
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    model_instance = model_instance.to(device)
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model_instance.train()
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(dataset):
            model_instance.zero_grad()
            all_adjs = data["adj"]
            all_feats = data["feats"]
            all_labels = data["label"]

            adj = Variable(data["adj"].float(), requires_grad=False)  # .cuda()
            V = Variable(data["feats"].float(), requires_grad=False)  # .cuda()
            label = Variable(data["label"].long())  # .cuda()

            ypred = model_instance(V.to(device), adj.to(device))
            predictions += ypred.cpu().detach().numpy().tolist()

            loss = model_instance.loss(ypred, label.to(device))
            loss.backward()
            nn.utils.clip_grad_norm(model_instance.parameters(), args.clip)
            optimizer.step()
            if batch_idx % 10 == 0:
                print("Batch {} optimized. Loss: {}" .format(batch_idx, loss.cpu().detach().numpy()))
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar("loss/avg_loss", avg_loss, epoch)
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)
        result = evaluate(
            dataset, model_instance, args, name="Train", max_num_examples=100)
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model_instance, args, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model_instance, args, testing=True, name="Test")
            test_result["epoch"] = epoch
        if writer is not None:
            writer.add_scalar("acc/train_acc", result["acc"], epoch)
            writer.add_scalar("acc/val_acc", val_result["acc"], epoch)
            writer.add_scalar("loss/best_val_loss", best_val_result["loss"], epoch)
            if test_dataset is not None:
                writer.add_scalar("acc/test_acc", test_result["acc"], epoch)

        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])
        if epoch %10 == 0:
            filename = io_utils.create_filename(args.ckptdir, args, False, epoch)
            torch.save(model_instance.state_dict(), filename)
    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    plt.figure()
    plt.plot(train_epochs, math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
        plt.legend(["train", "val", "test"])
    else:
        plt.plot(best_val_epochs, best_val_accs, "bo")
        plt.legend(["train", "val"])
    matplotlib.style.use("default")

    print("Shapes of \'all_adjs\', \'all_feats\', \'all_labels\':",
          all_adjs.shape,
          all_feats.shape,
          all_labels.shape, sep="\n")

    cg_data = {
        "adj": all_adjs,
        "feat": all_feats,
        "label": all_labels,
        "pred": np.expand_dims(predictions, axis=0),
        "train_idx": list(range(len(dataset))),
    }
    io_utils.save_checkpoint(model_instance, optimizer, args, num_epochs=-1,
                             cg_dict=cg_data)
    return model_instance, val_accs


def evaluate(dataset, model, args, name="Validation", testing=False, max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False)  # .cuda()
        h0 = Variable(data["feats"].float())  # .cuda()
        labels.append(data["label"].long().numpy())

        # TODO: fix the evaluate.
        ypred = model.forward(h0.to(device), adj.to(device))
        _, indices = torch.max(ypred, -1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels).squeeze()
    print("Label: ", labels.shape)
    preds = np.hstack(preds).squeeze()
    print("Predict: ", preds.shape)

    result = {
        # "prec": metrics.precision_score(labels, preds, average="macro"),
        # "recall": metrics.recall_score(labels, preds, average="macro"),
        "prec": metrics.precision_score(labels, preds, average="micro"),
        "recall": metrics.recall_score(labels, preds, average="micro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    if testing:
        print(metrics.classification_report(labels, preds, target_names=dataset.labels.keys()))
    print(name, " accuracy:", result["acc"])
    return result


class dummyArgs(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    random.seed(777)

    data_loader = FunsdDataLoader("./funsd_preprocess.pkl")
    data_loader_test = FunsdDataLoader("./funsd_preprocess_test.pkl", data_loader.labels)
    # set up the arguments
    args = dummyArgs()
    args.batch_size = 1
    args.bmname = None
    args.hidden_dim = 500
    args.dataset = "invoice"
    args.output_dim = len(data_loader.labels.keys())
    args.clip = True
    args.ckptdir = "ckpt"
    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)
    args.method = "GCN"
    args.name = "dummy name"
    args.num_epochs = 300
    args.train_ratio = 0.8
    args.test_ratio = 0.0
    args.gpu = torch.cuda.is_available()

    # data_loader = PerGraphNodePredDataLoader("../Invoice_k_fold/save_features/all/input_features.pickle")

    i = 0
    feature_dim = data_loader[i]['feats'].shape[-1]
    n_labels = args.output_dim
    n_edges = data_loader[i]['adj'].shape[-1]

    graph_kv = FUNSDModelConfig1(input_dim=feature_dim,
                                 output_dim=n_labels,
                                 num_edges=n_edges)
    graph_kv.to(device)
    graphs = data_loader
    test_graphs = data_loader_test
    indices = list(range(len(graphs)))
    random.shuffle(indices)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        # train_graphs = graphs[indices[:train_idx]]
        train_graphs = [graphs[i] for i in indices[:train_idx]]
        # val_graphs = graphs[indices[train_idx:test_idx]]
        val_graphs = [graphs[i] for i in indices[train_idx:test_idx]]
        # test_graphs = graphs[indices[test_idx:]]
        test_graphs = [graphs[i] for i in indices[test_idx:]]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = [graphs[i] for i in indices[:train_idx]]
        val_graphs = [graphs[i] for i in indices[train_idx:]]
    print(
        "Num training graphs: ",
        len(train_graphs),
        "; Num validation graphs: ",
        len(val_graphs),
        "; Num testing graphs: ",
        len(test_graphs),
    )

    train(train_graphs,
          model_instance=graph_kv,
          args=args,
          same_feat=True,
          val_dataset=val_graphs,
          test_dataset=test_graphs,
          writer=None,
          mask_nodes=True,
          )

    print("Finished\n\n")
    # pass
