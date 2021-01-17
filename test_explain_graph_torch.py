""" explainer_main.py of Cinnamon graph-kv version.

     Main user interface for the explainer module.
"""
import argparse
import os

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import pickle
import shutil
import torch
from torch.autograd import Variable

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import numpy as np


from explainer import explain_cinnamon as explain
# ExplainerMultiEdges()
# ExplainMultiEdgesModule()

from train_cinnamon import PerGraphNodePredDataLoader
from models.graph_kv_core import RobustFilterGraphCNNConfig1
from kv.graphkv_torch import GraphKV_Torch
from kv.graphkv_torch.utils.data_encoder import InputEncoder

import json

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    return data

class GraphKV_Torch_old(torch.nn.Module):
    """ Interface for GraphKV PyTorch version """
    def __init__(self, checkpoint_path, device='cpu'):
        """
        Input:
            checkpoint_path: Path to checkpoint after training
            device: gpu or cpu

        """
        super(GraphKV_Torch, self).__init__()
        self.load_checkpoint(checkpoint_path)
        self.to(device)


    def to(self, device):
        if device == "gpu":
            device = "cuda"

        self.device = device
        super(GraphKV_Torch, self).to(device)



    def load_checkpoint(self, checkpoint_path):
        # Load thing that we will need
        checkpoint = torch.load(checkpoint_path)
        self.ind2cls = checkpoint['ind2cls']

        self.load_model(checkpoint)

    def load_model(self, checkpoint: dict):
        # Load Hyper Parameter
        hparams = checkpoint['hparams']

        n_in, n_classes = hparams['n_in'], hparams['n_classes']
        print(n_in, n_classes)
        n_edges, net_size = hparams['n_edges'], hparams['net_size']

        # Build model core
        self.core = RobustFilterGraphCNNConfig1(n_in,
                n_classes, n_edges, net_size)

        # Load weight
        self.load_state_dict(checkpoint['state_dict'])

        self.normalize_vertex = hparams['normalize_vertex']
        self.core.eval()


    def forward(self, x, adj):
        return self.core(x, adj)

    def predict(self, x, adj):
        x = x.unsqueeze(0)
        adj = adj.unsqueeze(0)
        logit = self.forward(x, adj)
        _, indices = torch.max(logit, -1)
        preds = indices.cpu().data.numpy()
        return preds



def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        dataset="syn1",
        opt="adam",
        opt_scheduler="none",
        cuda="0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=100,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
    )
    return parser.parse_args()


class dummyArgs(object):
    def __init__(self):
        pass


def forward_pred(dataset, model_instance):
    """

    :param dataset:  Data_loader.
    :param model:    Pytorch model instance.
    :return:
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_instance = model_instance.to(device)
    model_instance.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False)  # .cuda()
        h0 = Variable(data["feats"].float())  # .cuda()
        labels.append(data["label"].long().numpy())

        # TODO: fix the evaluate.
        ypred = model_instance.forward(h0.to(device), adj.to(device))
        # ypred = model(V.to(device), adj.to(device))

        _, indices = torch.max(ypred, -1)
        preds.append(indices.cpu().data.numpy())

    return preds, labels


if __name__ == "__main__":
    # Load a configuration
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    parser.add_argument("-i", "--graph_idx", type=int, default=7,
                        help="Graph sample index.")
    parser.add_argument('--input', help='Path to input_features.pickle')
    parser.add_argument('--corpus', help='Path to corpus.json')
    parser.add_argument('--classes', help='Path to classes.json')

    args = parser.parse_args()

    prog_args = dummyArgs()
    data_loader = PerGraphNodePredDataLoader("./input_features.pickle", False)
    corpus = load_json(args.corpus)
    import pdb; pdb.set_trace
    # data_loader = PerGraphNodePredDataLoader("../Invoice_k_fold/save_features/all/input_features.pickle")

    prog_args.batch_size = 1
    prog_args.bmname = None
    prog_args.hidden_dim = 500
    prog_args.dataset = "invoice"
    prog_args.output_dim = np.max(np.array(np.concatenate(data_loader.labels, axis=0))) + 1
    prog_args.clip = True
    prog_args.method = "GCN"
    prog_args.name = "dummy name"
    # prog_args.num_epochs = 20
    prog_args.num_epochs = 2000
    prog_args.train_ratio = 0.8
    prog_args.test_ratio = 0.1
    prog_args.gpu = torch.cuda.is_available()
    prog_args.cuda = "2"

    prog_args.writer = None
    prog_args.logdir = os.path.join(os.getcwd(), "explain_log_new")
    prog_args.explainer_suffix = "explained_"

    # Load a model checkpoint
    input_dim = data_loader[0]['feats'].cpu().detach().numpy().shape[2]

    # num_classes = len(cg_dict["pred"][0][0][0])
    inp_classes = load_json(args.classes)
    num_classes =  2 * len(load_json(args.classes)) - 1 # Hot fix for SMBC right now: Num class * 2 + other

    print("input dim: ", input_dim, "; num classes: ", num_classes)

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # Configure the logging directory
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
           print('Removing existing log dir: ', path)
           if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": sys.exit(1)
           shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    i = 0

    # infer_graphs = [data_loader[i] for i in range(10)]
    feature_dim = data_loader[i]['feats'].shape[-1]
    n_labels = prog_args.output_dim
    n_edges = data_loader[i]['adj'].shape[-1]
    # model = GraphKV_Torch('./model_checkpoint_16_07_2020 16_29_37.cptk', 'gpu')
    model = GraphKV_Torch('./latest_model.cptk', 'gpu',
                          InputEncoder(corpus),
                          classes=inp_classes)

    '''
    RobustFilterGraphCNNConfig1(input_dim=feature_dim,
                                        output_dim=n_labels,
                                        num_edges=n_edges)
                                        '''
    model = model.core

    if prog_args.gpu:
        model = model.cuda()
    # load state_dict (obtained by model.state_dict() when saving checkpoint)

    # Create explainer
    # TODO: Choose graph_idx.

    prog_args.mask_act = "sigmoid"  # "ReLU"
    prog_args.opt = 'adam'
    prog_args.lr = 0.003
    prog_args.opt_scheduler = 'none'
    for i in range(len(data_loader)):
        prog_args.graph_idx = i
        explainer = explain.ExplainerMultiEdges(
            model=model,
            train_idx= 0, # cg_dict["train_idx"],
            args=prog_args,
            writer=writer,
            print_training=True,
            data_loader=data_loader
            # graph_idx=prog_args.graph_idx,
        )

        # TODO: API should definitely be cleaner
        # Let's define exactly which modes we support
        # We could even move each mode to a different method (even file)

        # TODO:
        print(data_loader.labels[prog_args.graph_idx])
        prog_args.explain_node = [j for j in range(len(data_loader.labels[prog_args.graph_idx]))
                                        if data_loader.labels[prog_args.graph_idx][j] > 0]
        prog_args.multinode_class = 1

        if prog_args.multinode_class >= 0:
            # print(cg_dict["label"])
            # only run for nodes with label specified by multinode_class
            # labels = cg_dict["label"][0]  # already numpy matrix
            print(
                "Node indices for label ",
                prog_args.multinode_class,
                " : ",
                # node_indices,
                prog_args.explain_node
            )
            explainer.explain_nodes(node_indices=prog_args.explain_node,
                                    corpus=corpus,
                                    graph_idx=prog_args.graph_idx)

