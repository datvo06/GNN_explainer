from __future__ import print_function, unicode_literals, division
import torch, pickle, time, matplotlib
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset


import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import utils.math_utils as math_utils
import utils.io_utils as io_utils

from utils.pickle_related import read_pickle
from models_cinnamon import RobustFilterGraphCNNConfig1

import random
import sys
# import shutil
__author__ = "Marc"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PerGraphNodePredDataLoader(Dataset):
    def __init__(self, dini_pickle_fp, transpose=True):
        inp_dict = pickle.load(open(dini_pickle_fp, 'rb'))
        self.inp_fps = inp_dict['file_paths']
        self.inp_adj = inp_dict['HeuristicGraphAdjMat']
        self.inp_bow = inp_dict['FormBowFeature']
        self.inp_cod = inp_dict['TextlineCoordinateFeature']
        self.labels = inp_dict['labels']
        self.transpose = transpose

    def __len__(self):
        # TODO: features.shape = (N_Graph, Node, Feature)
        return len(self.inp_fps)

    def getitem(self, idx):
        return {
            "adj": torch.Tensor(self.inp_adj[idx]).unsqueeze(0) if\
                    self.transpose else torch.Tensor(self.inp_adj[idx]).unsqueeze(0),
            "feats": torch.Tensor(np.concatenate((self.inp_bow[idx], self.inp_cod[idx]), -1)).unsqueeze(0),
            "label": torch.Tensor(self.labels[idx]).unsqueeze(0)
        }

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            list_idx = idx[:]
            return_lists = [self.getitem(idx) for idx in list_idx]
        else:
            return_lists = self.getitem(idx)

def train(dataset, model_instance, args, same_feat=True,
          val_dataset=None,
          test_dataset=None,
          writer=None,
          mask_nodes=True,
          ):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_instance.parameters()), lr=0.001
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
            test_result = evaluate(test_dataset, model_instance, args, name="Test")
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
    plt.savefig(io_utils.gen_train_plt_name(args), dpi=600)
    plt.close()
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


def evaluate(dataset, model, args, name="Validation", max_num_examples=None):
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
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


class dummyArgs(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    random.seed(777)

    # first arg: Train input_features
    # 2nd arg: Val, if no 2nd arg => split train to make val
    data_loader_train = PerGraphNodePredDataLoader(sys.argv[1])
    # set up the arguments
    args = dummyArgs()
    args.batch_size = 1
    args.bmname = None
    args.hidden_dim = 500
    args.dataset = "invoice"
    args.output_dim = np.max(np.array(np.concatenate(
        data_loader_train.labels, axis=0))) + 1
    args.clip = True
    args.ckptdir = "ckpt"
    args.method = "GCN"
    args.name = "dummy name"
    args.num_epochs = 200
    args.train_ratio = 0.8
    args.test_ratio = 0.1
    args.gpu = torch.cuda.is_available()

    i = 0
    feature_dim = data_loader_train[i]['feats'].shape[-1]
    n_labels = args.output_dim
    n_edges = data_loader_train[i]['adj'].shape[-1]

    graph_kv = RobustFilterGraphCNNConfig1(input_dim=feature_dim,
                                           output_dim=n_labels,
                                           num_edges=n_edges)
    graph_kv.to(device)
    graphs = data_loader_train
    indices = list(range(len(graphs)))
    random.shuffle(indices)
    if len(sys.argv) < 3:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))
        train_graphs = [graphs[i] for i in indices[:train_idx]]
        val_graphs = [graphs[i] for i in indices[train_idx:test_idx]]
        test_graphs = [graphs[i] for i in indices[test_idx:]]
    else:
        data_loader_val = PerGraphNodePredDataLoader(sys.argv[2])
        val_graphs = data_loader_val
        test_graphs = data_loader_val
        train_graphs = graphs[indices]
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
