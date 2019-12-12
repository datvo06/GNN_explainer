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

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import numpy as np


from explainer import explain_cinnamon as explain
# ExplainerMultiEdges()
# ExplainMultiEdgesModule()

from train_cinnamon import PerGraphNodePredDataLoader
from models_cinnamon import RobustFilterGraphCNNConfig1


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


if __name__ == "__main__":
    # Load a configuration
    # prog_args = arg_parse()

    prog_args = dummyArgs()
    data_loader = PerGraphNodePredDataLoader("./Invoice_data/input_features.pickle")
    # data_loader = PerGraphNodePredDataLoader("../Invoice_k_fold/save_features/all/input_features.pickle")
    prog_args.batch_size = 1
    prog_args.bmname = None
    prog_args.hidden_dim = 500
    prog_args.dataset = "invoice"
    prog_args.output_dim = np.max(np.array(np.concatenate(data_loader.labels, axis=0))) + 1
    prog_args.clip = True
    prog_args.ckptdir = "ckpt"
    prog_args.method = "GCN"
    prog_args.name = "dummy name"
    #prog_args.num_epochs = 200
    prog_args.num_epochs = 2
    prog_args.train_ratio = 0.8
    prog_args.test_ratio = 0.1
    prog_args.gpu = torch.cuda.is_available()
    prog_args.cuda = "2"

    prog_args.writer = None
    prog_args.logdir = os.path.join(os.getcwd(), "explain_log")
    prog_args.explainer_suffix = "_explained"

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"] # get computation graph
    input_dim = cg_dict["feat"].cpu().detach().numpy().shape[2]

    # cg_dict["pred"][0][sample_idx][textline_idx] = kv last layer output.
    num_classes = len(cg_dict["pred"][0][0][0])

    print("Loaded model from {}".format(prog_args.ckptdir))
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

    '''
    # Determine explainer mode
    graph_mode = (
        prog_args.graph_mode
        or prog_args.multigraph_class >= 0
        or prog_args.graph_idx >= 0
    )

    # build model
    print("Method: ", prog_args.method)
    if graph_mode:
        # Explain Graph prediction
        model = models.GcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    else:
        if prog_args.dataset == "ppi_essential":
            # class weight in CE loss for handling imbalanced label classes
            prog_args.loss_weight = torch.tensor([1.0, 5.0], dtype=torch.float).cuda()
        # Explain Node prediction
        model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
        '''

    i = 0
    feature_dim = data_loader[i]['feats'].shape[-1]
    n_labels = prog_args.output_dim
    n_edges = data_loader[i]['adj'].shape[-1]
    model = RobustFilterGraphCNNConfig1(input_dim=feature_dim,
                                        output_dim=n_labels,
                                        num_edges=n_edges)

    if prog_args.gpu:
        model = model.cuda()
    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    model.load_state_dict(ckpt["model_state"])

    # Create explainer
    # TODO: Choose graph_idx.
    prog_args.graph_idx = 0
    prog_args.mask_act = "sigmoid"  # "ReLU"
    prog_args.opt = 'adam'
    prog_args.lr = 0.0001
    prog_args.opt_scheduler = 'none'

    explainer = explain.ExplainerMultiEdges(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        writer=writer,
        print_training=True,
        graph_idx=prog_args.graph_idx,
    )

    # TODO: API should definitely be cleaner
    # Let's define exactly which modes we support
    # We could even move each mode to a different method (even file)

    # TODO:
    prog_args.explain_node = [i for i in range(len(cg_dict['label'][prog_args.graph_idx])) if cg_dict['label'][prog_args.graph_idx][i] > 0]
    prog_args.multinode_class = 1

    # explainer.explain(prog_args.explain_node, unconstrained=False)
    explainer.explain_nodes_gnn_stats(prog_args.explain_node, prog_args)

    if prog_args.multinode_class >= 0:
        print(cg_dict["label"])
        # only run for nodes with label specified by multinode_class
        labels = cg_dict["label"][0]  # already numpy matrix

        node_indices = []
        for i, l in enumerate(labels):
            if len(node_indices) > 4:
                break
            if l == prog_args.multinode_class:
                node_indices.append(i)
        print(
            "Node indices for label ",
            prog_args.multinode_class,
            " : ",
            node_indices,
        )
        explainer.explain_nodes(node_indices, prog_args)

    else:
        # explain a set of nodes
        masked_adj = explainer.explain_nodes_gnn_stats(
            range(400, 700, 5), prog_args
        )

