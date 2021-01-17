""" explain.py

    Implementation of the explainer.
"""

import math
import time
import os

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorboardX.utils

import torch
import torch.nn as nn
from torch.autograd import Variable

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils

from utils.draw_utils import visualize_graph, visualize_graph_text
import cv2


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

class ExplainerMultiEdges:
    def __init__(
        self, model,
        # adj, feat, label, pred,
        data_loader,
        args, writer=None, print_training=True,
        train_idx=0,
        # graph_idx=False,
        use_unsqueeze=True
    ):
        self.train_idx = 0
        self.model = model
        self.model.eval()
        # self.adj = adj
        # self.feat = feat
        # self.label = label
        # self.pred = pred
        self.data_loader = data_loader

        # self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.print_training = print_training
        self.use_unsqueeze = use_unsqueeze


    # Main method
    def explain(
        self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model="exp"
    ):
        """Explain a single node prediction
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # index of the query node in the new adj
        # we always use all of the nodes since we have self-attention
        # All of the current stuffs included batch, just squeeze...
        node_idx_new = node_idx
        # adj = self.adj.squeeze() # N N L
        # feat = self.feat.squeeze() # N F
        # label = self.label.squeeze().long()

        adj = self.data_loader[graph_idx]['adj'].squeeze() # N N L
        feat = self.data_loader[graph_idx]['feats'].squeeze() # N F
        label = self.data_loader[graph_idx]['label'].squeeze().long()

        # print(sub_label)
        x     = torch.tensor(feat, requires_grad=True, dtype=torch.float).to(device)
        '''
        self.pred: matrix, first row is a n_graphs-dimensional vector, each of which contain a list storing the Graph features
        '''

        # TODO: fix the evaluate.
        # pred_label = torch.Tensor(np.argmax(np.array(self.pred[0, graph_idx]), axis=1)).to(device)

        adj = Variable(adj.float(), requires_grad=False)  # .cuda()
        h0 = Variable(feat.float())  # .cuda()

        ypred = self.model.forward(
            h0.to(device).unsqueeze(0) if self.use_unsqueeze else h0.to(device),
            adj.to(device).unsqueeze(0) if self.use_unsqueeze else adj.to(device))
        _, indices = torch.max(ypred, -1)
        pred_label = indices.to(device)

        explainer = ExplainMultiEdgesModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            use_unsqueeze=self.use_unsqueeze,
            # graph_idx=graph_idx,
            # graph_mode=self.graph_mode,
        )
        # The explainer is used to create and maintain the masks
        # otherwise it will not interferes with model's prediction
        # This means that we may use basically anything
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()

        # Starting the optimization progress
        # As stated in the paper, the mask is found by
        # an optimization process
        explainer.train()
        begin_time = time.time()
        print("node index new: ", node_idx_new)
        '''
        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            ypred = explainer(node_idx_new, unconstrained=unconstrained)
            loss = explainer.loss(ypred, pred_label, graph_idx, node_idx_new, epoch)
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            mask_density = explainer.mask_density()
            if self.print_training:
                if epoch % 10 == 0:
                    print(
                        "epoch: ", epoch,
                        "; loss: ", loss.item(),
                        "; mask density: ", mask_density.item(),
                        # "; pred: ",
                        # ypred,
                    )
            single_subgraph_label = label.squeeze()

            if self.writer is not None:
                self.writer.add_scalar("mask/density", mask_density, epoch)
                self.writer.add_scalar(
                    "optimization/lr",
                    explainer.optimizer.param_groups[0]["lr"],
                    epoch,
                )
                if epoch % 25 == 0:
                    explainer.log_mask(epoch)
                    explainer.log_masked_adj(
                        node_idx_new, epoch, label=single_subgraph_label
                    )
                    explainer.log_adj_grad(
                        node_idx_new, pred_label, epoch, label=single_subgraph_label
                    )
        '''

        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            ypred = explainer.forward_per_node_features(
                node_idx_new, unconstrained=unconstrained)
            loss = explainer.loss_for_per_node_feats(
                ypred, pred_label, graph_idx, node_idx_new, epoch)
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            mask_density = explainer.mask_density()
            if self.print_training:
                if epoch % 10 == 0:
                    print(
                        "epoch: ", epoch,
                        "; loss: ", loss.item(),
                        "; mask density: ", mask_density.item(),
                        # "; pred: ",
                        # ypred,
                    )
            single_subgraph_label = label.squeeze()

            if self.writer is not None:
                self.writer.add_scalar("mask/node_feat_density", mask_density, epoch)
                self.writer.add_scalar(
                    "node_feat_optimization/lr",
                    explainer.optimizer.param_groups[0]["lr"],
                    epoch,
                )
                if epoch % 25 == 0:
                    explainer.log_mask(epoch)
                    explainer.log_masked_adj(
                        node_idx_new, epoch, label=single_subgraph_label
                    )
                    explainer.log_adj_grad(
                        node_idx_new, pred_label, epoch, label=single_subgraph_label
                    )



        print("finished training in ", time.time() - begin_time)
        masked_adj = (
            explainer._masked_adj().cpu().detach().numpy() * adj.squeeze().cpu().detach().numpy()
        )
        masked_node_feats = explainer.node_feat_mask.cpu().detach().numpy() * feat.squeeze().cpu().detach().numpy()
        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'node_idx_'+str(node_idx)+'_graph_idx_'+str(graph_idx)+'.npy')
        fname2 = 'masked_per_node_feat_' + io_utils.gen_explainer_prefix(
            self.args) + (
                'node_idx_'+str(node_idx)+'_graph_idx_'+str(graph_idx)+'.npy')
                # 'node_idx_'+str(node_idx)+'graph_idx_'+str(self.graph_idx)+'.npy')


        with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            print("Saved adjacency matrix to ", fname)

        with open(os.path.join(self.args.logdir, fname2), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_node_feats.copy()))
            print("Saved node feats matrix to ", fname2)

        return masked_adj, explainer.get_node_mask().squeeze().cpu().detach().numpy()


    # MASKED ADJ EXPLAINER
    def explain_nodes(self, node_indices,
                      # args, data_loader,
                      list_texts=None, corpus=None, graph_idx=0):
        """
        Explain nodes

        Args:

        :param node_indices : Indices of the nodes to be explained
        :param args         : Program arguments (mainly for logging paths)
        :param corpus       : String, the Bag of Words character correspond to the GCN input.
        :param data_loader  : The data_loader for PyTorch.
        :param graph_idx    : Index of the graph to explain the nodes from (if multiple).

        :return:

        """

        # TODO: Change Draw Function here.
        # pred_all, real_all, masked_adjs = [], [], []
        node_masks = []
        masked_adjs = []
        for i, node_idx in enumerate(node_indices):

            # Get explanations for each of the nodes
            masked_adj, node_feat_mask = self.explain(node_idx, graph_idx=graph_idx)
            # pred, real = self.make_pred_real(masked_adj, node_idx)
            masked_adjs.append(masked_adj)
            node_masks.append(node_feat_mask)
            # pred_all.append(pred)
            # real_all.append(real)

            coord = self.data_loader[graph_idx]['feats'][:, :, -4:].squeeze().cpu().detach().numpy()
            coord = coord * 1.1 - 0.1

            adj = self.data_loader[graph_idx]['adj'].squeeze().cpu().detach().numpy()

            label_y = self.data_loader[graph_idx]['label'].squeeze().cpu().detach().numpy()

            # Number of total Nodes/Textlines in this Graph.
            # N = bow.shape[0]

            if list_texts is None:
                bow = self.data_loader.inp_bow[graph_idx]
                image = visualize_graph(list_bows=bow,
                                        list_positions=(coord * 1000).astype(int),
                                        adj_mats=adj,
                                        node_labels=label_y,
                                        adj_importances=masked_adj,
                                        # node_importances=node_mask,
                                        bow_importances=node_feat_mask[:, :-4],
                                        word_list=corpus,
                                        cur_node_idx=node_idx
                                        )
            else:
                image = visualize_graph_text(list_texts,
                                        list_positions=(coord * 1000).astype(int),
                                        adj_mats=adj,
                                        node_labels=label_y,
                                        adj_importances=masked_adj,
                                        bow_importances=node_feat_mask[:, :-4],
                                        # node_importances=node_mask,
                                        word_list=corpus,
                                        cur_node_idx=node_idx
                                        )


            try: os.makedirs(os.path.join(self.args.logdir, "Graph_{}".format(graph_idx)))
            except FileExistsError: pass

            save_path = os.path.join(self.args.logdir, "Graph_{}/node_{}.png".format(graph_idx, node_idx))
            cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 5])

        return masked_adjs

    # MASKED NODE EXPLAINER
    def explain_nodes_feats(self, node_indices,
                      # args, data_loader,
                      list_texts=None, corpus=None, graph_idx=0):
        """
        Explain nodes

        Args:

        :param node_indices : Indices of the nodes to be explained
        :param args         : Program arguments (mainly for logging paths)
        :param corpus       : String, the Bag of Words character correspond to the GCN input.
        :param data_loader  : The data_loader for PyTorch.
        :param graph_idx    : Index of the graph to explain the nodes from (if multiple).

        :return:

        """

        # TODO: Change Draw Function here.
        # pred_all, real_all, masked_adjs = [], [], []
        node_masks = []
        masked_adjs = []
        for i, node_idx in enumerate(node_indices):

            # Get explanations for each of the nodes
            masked_adj, node_mask = self.explain(node_idx, graph_idx=graph_idx)
            # pred, real = self.make_pred_real(masked_adj, node_idx)
            masked_adjs.append(masked_adj)
            node_masks.append(node_mask)
            # pred_all.append(pred)
            # real_all.append(real)

            coord = self.data_loader[graph_idx]['feats'][:, :, -4:].squeeze().cpu().detach().numpy()
            print(coord.shape)
            coord = coord * 1.1 - 0.1

            adj = self.data_loader[graph_idx]['adj'].squeeze().cpu().detach().numpy()

            label_y = self.data_loader[graph_idx]['label'].squeeze().cpu().detach().numpy()

            # Number of total Nodes/Textlines in this Graph.
            # N = bow.shape[0]

            if list_texts is None:
                bow = self.data_loader.inp_bow[graph_idx]
                image = visualize_graph(list_bows=bow,
                                        list_positions=(coord * 1000).astype(int),
                                        adj_mats=adj,
                                        node_labels=label_y,
                                        adj_importances=masked_adj,
                                        # node_importances=node_mask,
                                        word_list=corpus,
                                        cur_node_idx=node_idx
                                        )
            else:
                image = visualize_graph_text(list_texts,
                                        list_positions=(coord * 1000).astype(int),
                                        adj_mats=adj,
                                        node_labels=label_y,
                                        adj_importances=masked_adj,
                                        # node_importances=node_mask,
                                        word_list=corpus,
                                        cur_node_idx=node_idx
                                        )


            try: os.makedirs(os.path.join(self.args.logdir, "Graph_{}".format(graph_idx)))
            except FileExistsError: pass

            save_path = os.path.join(self.args.logdir, "Graph_{}/node_{}.png".format(graph_idx, node_idx))
            cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 5])

        return masked_adjs



    def explain_nodes_gnn_stats(self, node_indices, args, graph_idx=0, model="exp"):
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, model=model)
            for node_idx in node_indices
        ]
        # pdb.set_trace()
        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []

        for i, idx in enumerate(node_indices):

            new_idx = idx
            pred, real = self.make_pred_real(masked_adjs[i], new_idx)
            pred_all.append(pred)
            real_all.append(real)

            '''
            feat = self.feat.squeeze()
            G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)


            denoised_feat = np.array([G.nodes[node]["feat"] for node in G.nodes()])
            denoised_adj = nx.to_numpy_matrix(G)
            graphs.append(G)
            feats.append(denoised_feat)
            adjs.append(denoised_adj)
            io_utils.log_graph(
                self.writer,
                G,
                "graph/{}_{}_{}".format(self.args.dataset, model, i),
                identify_self=True,
            )
            '''

        pred_all = np.concatenate((pred_all), axis=0)
        real_all = np.concatenate((real_all), axis=0)
        auc_all = roc_auc_score(real_all, pred_all)

        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")
        plt.close()

        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        with open("log/pr/auc_" + self.args.dataset + "_" + model + ".txt", "w") as f:
            f.write(
                "dataset: {}, model: {}, auc: {}\n".format(
                    self.args.dataset, "exp", str(auc_all)
                )
            )

        return masked_adjs

    # GRAPH EXPLAINER
    def explain_graphs(self, graph_indices):
        """
        Explain graphs.
        """
        masked_adjs = []

        for graph_idx in graph_indices:
            masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True)
            G_denoised = io_utils.denoise_graph(
                masked_adj,
                0,
                threshold_num=20,
                feat=self.feat[graph_idx],
                max_component=False,
            )
            label = self.label[graph_idx]
            io_utils.log_graph(
                self.writer,
                G_denoised,
                "graph/graphidx_{}_label={}".format(graph_idx, label),
                identify_self=False,
                nodecolor="feat",
            )
            masked_adjs.append(masked_adj)

            G_orig = io_utils.denoise_graph(
                self.adj[graph_idx],
                0,
                feat=self.feat[graph_idx],
                threshold=None,
                max_component=False,
            )

            io_utils.log_graph(
                self.writer,
                G_orig,
                "graph/graphidx_{}".format(graph_idx),
                identify_self=False,
                nodecolor="feat",
            )

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")

        return masked_adjs

    def log_representer(self, rep_val, sim_val, alpha, graph_idx=0):
        """ visualize output of representer instances. """
        rep_val = rep_val.cpu().detach().numpy()
        sim_val = sim_val.cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()
        sorted_rep = sorted(range(len(rep_val)), key=lambda k: rep_val[k])
        print(sorted_rep)
        topk = 5
        most_neg_idx = [sorted_rep[i] for i in range(topk)]
        most_pos_idx = [sorted_rep[-i - 1] for i in range(topk)]
        rep_idx = [most_pos_idx, most_neg_idx]

        if self.graph_mode:
            pred = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
            pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        # print(metrics.confusion_matrix(self.label[graph_idx][self.train_idx], pred))
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(5, 3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                print(
                    "node idx: ",
                    idx,
                    "; node label: ",
                    self.label[graph_idx][idx],
                    "; pred: ",
                    pred,
                )

                idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                    idx, graph_idx
                )
                G = nx.from_numpy_matrix(sub_adj)
                node_colors = [1 for i in range(G.number_of_nodes())]
                node_colors[idx_new] = 0
                # node_color='#336699',

                ax = plt.subplot(2, topk, i * topk + j + 1)
                nx.draw(
                    G,
                    pos=nx.spring_layout(G),
                    with_labels=True,
                    font_size=4,
                    node_color=node_colors,
                    cmap=plt.get_cmap("Set1"),
                    vmin=0,
                    vmax=8,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    width=0.5,
                    node_size=25,
                    alpha=0.7,
                )
                ax.xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image(
            "local/representer_neigh", tensorboardX.utils.figure_to_image(fig), 0
        )

    def representer(self):
        """
        experiment using representer theorem for finding supporting instances.
        https://papers.nips.cc/paper/8141-representer-point-selection-for-explaining-deep-neural-networks.pdf
        """
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds, _ = self.model(x, adj)
        preds.retain_grad()
        self.embedding = self.model.embedding_tensor
        loss = self.model.loss(preds, label)
        loss.backward()
        self.preds_grad = preds.grad
        pred_idx = np.expand_dims(np.argmax(self.pred, axis=2), axis=2)
        pred_idx = torch.LongTensor(pred_idx)
        if self.args.gpu:
            pred_idx = pred_idx.cuda()
        self.alpha = self.preds_grad


    # Utilities
    def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    def align(
        self, ref_feat, ref_adj, ref_node_idx, curr_feat, curr_adj, curr_node_idx, args
    ):
        """ Tries to find an alignment between two graphs.
        """
        ref_adj = torch.FloatTensor(ref_adj)
        curr_adj = torch.FloatTensor(curr_adj)

        ref_feat = torch.FloatTensor(ref_feat)
        curr_feat = torch.FloatTensor(curr_feat)

        P = nn.Parameter(torch.FloatTensor(ref_adj.shape[0], curr_adj.shape[0]))
        with torch.no_grad():
            nn.init.constant_(P, 1.0 / ref_adj.shape[0])
            P[ref_node_idx, :] = 0.0
            P[:, curr_node_idx] = 0.0
            P[ref_node_idx, curr_node_idx] = 1.0
        opt = torch.optim.Adam([P], lr=0.01, betas=(0.5, 0.999))
        for i in range(args.align_steps):
            opt.zero_grad()
            feat_loss = torch.norm(P @ curr_feat - ref_feat)

            aligned_adj = P @ curr_adj @ torch.transpose(P, 0, 1)
            align_loss = torch.norm(aligned_adj - ref_adj)
            loss = feat_loss + align_loss
            loss.backward()  # Calculate gradients
            self.writer.add_scalar("optimization/align_loss", loss, i)
            print("iter: ", i, "; loss: ", loss)
            opt.step()

        return P, aligned_adj, P @ curr_feat

    def make_pred_real(self, adj, start):
        # house graph
        if self.args.dataset == "syn1" or self.args.dataset == "syn2":
            # num_pred = max(G.number_of_edges(), 6)
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start][start + 3] > 0:
                real[start][start + 3] = 10
            if real[start][start + 4] > 0:
                real[start][start + 4] = 10
            if real[start + 1][start + 4]:
                real[start + 1][start + 4] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        # cycle graph
        elif self.args.dataset == "syn4":
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            # pdb.set_trace()
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start + 3][start + 4] > 0:
                real[start + 3][start + 4] = 10
            if real[start + 4][start + 5] > 0:
                real[start + 4][start + 5] = 10
            if real[start][start + 5]:
                real[start][start + 5] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        return pred, real

    # NODE EXPLAINER
    def explain_nodes(self, node_indices,
                      # args, data_loader,
                      list_texts=None, corpus=None, graph_idx=0):
        """
        Explain nodes

        Args:

        :param node_indices : Indices of the nodes to be explained
        :param args         : Program arguments (mainly for logging paths)
        :param corpus       : String, the Bag of Words character correspond to the GCN input.
        :param data_loader  : The data_loader for PyTorch.
        :param graph_idx    : Index of the graph to explain the nodes from (if multiple).

        :return:

        """

        # TODO: Change Draw Function here.
        # pred_all, real_all, masked_adjs = [], [], []
        node_masks = []
        masked_adjs = []
        for i, node_idx in enumerate(node_indices):

            # Get explanations for each of the nodes
            masked_adj, node_mask = self.explain(node_idx, graph_idx=graph_idx)
            # pred, real = self.make_pred_real(masked_adj, node_idx)
            masked_adjs.append(masked_adj)
            node_masks.append(node_mask)
            # pred_all.append(pred)
            # real_all.append(real)

            coord = self.data_loader[graph_idx]['feats'][:, :, -4:].squeeze().cpu().detach().numpy()
            print(coord.shape)
            coord = coord * 1.1 - 0.1

            adj = self.data_loader[graph_idx]['adj'].squeeze().cpu().detach().numpy()

            label_y = self.data_loader[graph_idx]['label'].squeeze().cpu().detach().numpy()

            # Number of total Nodes/Textlines in this Graph.
            # N = bow.shape[0]

            if list_texts is None:
                bow = self.data_loader.inp_bow[graph_idx]
                image = visualize_graph(list_bows=bow,
                                        list_positions=(coord * 1000).astype(int),
                                        adj_mats=adj,
                                        node_labels=label_y,
                                        adj_importances=masked_adj,
                                        # node_importances=node_mask,
                                        word_list=corpus,
                                        cur_node_idx=node_idx
                                        )
            else:
                image = visualize_graph_text(list_texts,
                                        list_positions=(coord * 1000).astype(int),
                                        adj_mats=adj,
                                        node_labels=label_y,
                                        adj_importances=masked_adj,
                                        # node_importances=node_mask,
                                        word_list=corpus,
                                        cur_node_idx=node_idx
                                        )


            try: os.makedirs(os.path.join(self.args.logdir, "Graph_{}".format(graph_idx)))
            except FileExistsError: pass

            save_path = os.path.join(self.args.logdir, "Graph_{}/node_{}.png".format(graph_idx, node_idx))
            cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 5])

        '''
        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat, _, _ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat, _, _ = self.extract_neighborhood(curr_idx)

        # Thresh hold the graph
        G_ref = io_utils.denoise_graph(
                ref_adj.cpu().detach().numpy(),
                new_ref_idx, ref_feat.cpu().detach().numpy(), threshold=0.1)
        denoised_ref_feat = np.array(
            [G_ref.nodes[node]["feat"] for node in G_ref.nodes()]
        )
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(
            curr_adj, new_curr_idx, curr_feat, threshold=0.1
        )
        denoised_curr_feat = np.array(
            [G_curr.nodes[node]["feat"] for node in G_curr.nodes()]
        )
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(
            denoised_ref_feat,
            denoised_ref_adj,
            ref_node_idx,
            denoised_curr_feat,
            denoised_curr_adj,
            curr_node_idx,
            args=args,
        )
        io_utils.log_matrix(self.writer, P, "align/P", 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, "align/ref")
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, "align/before")

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        print("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")

        # io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)
        '''

        return masked_adjs



class ExplainMultiEdgesModule(nn.Module):
    def __init__(
        self,
        adj,
        x,
        model,
        label,
        args,
        graph_idx=0,
        writer=None,
        use_sigmoid=True,
        graph_mode=False,
        use_unsqueeze=True
    ):
        """
        Args:
            adj: the adjacency matrix (NxN)
            x: the node features (NxD)
            label: the node labels (N)
        """
        super(ExplainMultiEdgesModule, self).__init__()

        adj = adj.squeeze()
        x = x.squeeze()
        assert (len(list(adj.size())) == 3), "Adj mush be NxNxL"
        assert len(list(x.size())) == 2, "x must be NxF"

        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode
        self.criterion = torch.nn.CrossEntropyLoss()

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        num_edges = adj.size()[-1]
        # First, init the edge mask and bias with normals
        # This is where we modify them
        # TODO: Make mask_bias cleaner.
        self.use_unsqueeze = use_unsqueeze
        self.args.mask_bias=True

        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, num_edges, init_strategy=init_strategy
        )
        self.node_mask, self.node_mask_bias = self.construct_node_mask(
            num_nodes, init_strategy=init_strategy
        )

        self.node_feat_mask, self.node_feat_mask_bias = self.construct_node_feat_mask(
            num_nodes, x.size(-1)
        )

        # The feature mask is used to highlight the important features
        self.feat_mask = self.construct_feat_mask(
            x.size(-1), init_strategy="constant")
        params = [self.mask, self.feat_mask]

        if self.mask_bias is not None:
            params.append(self.mask_bias)

        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes, num_edges) - torch.eye(num_nodes).unsqueeze(-1)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        # Normally, optimizer is just sgd, scheduler can be used to dimnish the
        # gradients
        self.scheduler, self.optimizer =\
            train_utils.build_optimizer(args, params)

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        """
        Args:
            feat_dim: an integer (D)
        """
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_node_mask(self, num_nodes, init_strategy="normal"):
        """
        Args:
            num_node: number of nodes
        """
        mask = nn.Parameter(torch.FloatTensor(num_nodes))
        mask_bias = nn.Parameter(torch.FloatTensor(num_nodes))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
                mask_bias.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                nn.init.constant_(mask_bias, 0.0)
        return mask, mask_bias

    def construct_node_feat_mask(self, num_nodes, feat_dim,
                                 init_strategy="normal", const_val=1.0):
        """
        Args:
            num_nodes: number of nodes (N)
            feat_dim: feature dimention (F)
        """
        mask = nn.Parameter(torch.FloatTensor(num_nodes, feat_dim))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes,
                                                       feat_dim))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def construct_edge_mask(self, num_nodes,
                            num_edges,
                            init_strategy="normal", const_val=1.0):
        """
        Args:
            num_nodes: number of nodes (N)
        """
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_edges))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes,
                                                       num_nodes,
                                                       num_edges))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias


    def _masked_feat(self, detach=False):
        if not detach:
            mask = self.node_feat_mask
            mask_bias = self.node_feat_mask_bias
        else:
            mask = self.node_feat_mask.detach()
            mask_bias = self.node_feat_mask_bias.detach()
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(mask)
        x = self.x.cuda() if self.args.gpu else self.x
        masked_x = x * mask
        if self.args.mask_bias:
            bias = nn.ReLU6()(mask_bias * 6) / 6
            masked_x += bias
        return masked_x


    def _masked_adj(self, detach=False):
        if not detach:
            mask = self.mask
            mask_bias = self.mask_bias
        else:
            mask = self.mask.detach()
            mask_bias = self.mask_bias.detach()
        sym_mask = mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(sym_mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(sym_mask)
        sym_mask = (sym_mask + sym_mask.transpose(0, 1)) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (mask_bias + mask_bias.transpose(0, 1)) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.transpose(0, 1)) / 2
        return masked_adj * self.diag_mask

    def get_node_mask(self):
        sym_mask = self.node_mask.unsqueeze(-1) + self.node_mask_bias.unsqueeze(-1)
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(sym_mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(sym_mask)
        return sym_mask


    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum


    def forward_per_node_features(self, node_idx,
                                  unconstrained=False):
        x = self._masked_feat()
        masked_adj = self._masked_adj()
        # masked_adj = self._masked_adj(detach=True)
        ypred = self.model(
            x.unsqueeze(0) if self.use_unsqueeze else x,
            masked_adj.unsqueeze(0) if self.use_unsqueeze else masked_adj)
        return ypred


    def forward(self, node_idx, unconstrained=False, mask_features=True,
                marginalize=False):
        """
        Args:
            node_idx: the chosen node's label to be explained
        """
        x = self.x.cuda() if self.args.gpu else self.x
        '''
        print(" x size: ", x.size())
        print(" node mask size: ", self.get_node_mask().size())
        x = self.get_node_mask() * x # Use boardcasting
        '''

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.transpose(0, 1)) / 2, 0) * self.diag_mask
            )
        else:
            self.masked_adj = self._masked_adj()
            if mask_features:
                feat_mask = (
                    torch.sigmoid(self.feat_mask)
                    if self.use_sigmoid
                    else self.feat_mask
                )
                if marginalize:
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x * feat_mask

        ypred = self.model(x.unsqueeze(0) if self.use_unsqueeze else x,
            self.masked_adj.unsqueeze(0) if self.use_unsqueeze else self.masked_adj)
        ''' # We dont use this stuff.
        if self.graph_mode:
            res = nn.Softmax(dim=0)(ypred[0])
        else:
            node_pred = ypred[self.graph_idx, node_idx, :]
            res = nn.Softmax(dim=0)(node_pred)
        '''
        return ypred

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss_consistency(self, pred, pred_label, node_idx):
        """
        Args:
            :param pred: prediction made by the current model the current mask Nxself.output_dim
            :param pred_label: prediction made by the model without the mask N
            :param node_idx: the node ids used for calculation
        :return:
        """
        # print("The previous model output: ", pred.view(-1, self.model.output_dim)[node_idx].size())
        # print(pred_label[node_idx].unsqueeze(-1).size())
        # input()
        return self.criterion(pred.view(-1, self.model.output_dim)[node_idx].unsqueeze(0),
                              pred_label.view(-1)[node_idx].unsqueeze(-1).long())

    def loss_for_per_node_feats(
            self, pred, pred_label, graph_idx, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        '''
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            # pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[graph_idx].squeeze()[node_idx]
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
            '''
        pred_loss = self.loss_consistency(pred, pred_label, node_idx)
        # size
        feat_mask = self.node_feat_mask
        if self.mask_act == "sigmoid":
            feat_mask = torch.sigmoid(feat_mask)
        elif self.mask_act == "ReLU":
            feat_mask = nn.ReLU()(feat_mask)
        # Size loss will make the mask as small as possible
        # in conjunction with the CE bellow, it will draw the mask towards
        # 0, unless the prediction loss pull it back
        # size_loss = self.coeffs["size"] * (torch.sum(mask) + torch.sum(self.get_node_mask())*mask.size()[1])
        feat_size_loss = self.coeffs["size"] * (torch.sum(feat_mask))

        # entropy
        # This loss is designed to keep the mask as separated (either dense or
        # sparse as possible
        # if mask element ~ 1 i.e, 0.99: loss = ~0 - ~0 = 0
        # if mask element ~0 i.e, 0.01: loss = ~0 - 0 = 0
        # else, if mask element 0.5, loss = -log(0.5) = log(2) (maximum value)
        '''
        node_mask = self.get_node_mask()
        node_mask_ent = -node_mask*torch.log(node_mask) - (1 - node_mask) * torch.log(1- node_mask)
        '''

        feat_mask_ent = -feat_mask * torch.log(feat_mask) -\
            (1 - feat_mask) * torch.log(1 - feat_mask)
        feat_mask_ent_loss = self.coeffs["ent"] * (torch.mean(feat_mask_ent))#  + torch.mean(node_mask_ent))

        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        # Size loss will make the mask as small as possible
        # in conjunction with the CE bellow, it will draw the mask towards
        # 0, unless the prediction loss pull it back
        # size_loss = self.coeffs["size"] * (torch.sum(mask) + torch.sum(self.get_node_mask())*mask.size()[1])
        size_loss = self.coeffs["size"] * (torch.sum(mask))
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * (torch.mean(mask_ent))#  + torch.mean(node_mask_ent))


        loss = pred_loss + feat_size_loss + feat_mask_ent_loss + mask_ent_loss + size_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/node_feat_size_loss",
                                   feat_size_loss, epoch)
            self.writer.add_scalar("optimization/node_feat_mask_ent_loss",
                                   feat_mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/node_feat_mask_ent_loss", feat_mask_ent_loss,
                epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/node_feat_pred_loss",
                                   pred_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss


    def loss(self, pred, pred_label, graph_idx, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        '''
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            # pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            gt_label_node = self.label if self.graph_mode else self.label[graph_idx].squeeze()[node_idx]
            logit = pred[gt_label_node]
            pred_loss = -torch.log(logit)
            '''
        pred_loss = self.loss_consistency(pred, pred_label, node_idx)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        # Size loss will make the mask as small as possible
        # in conjunction with the CE bellow, it will draw the mask towards
        # 0, unless the prediction loss pull it back
        # size_loss = self.coeffs["size"] * (torch.sum(mask) + torch.sum(self.get_node_mask())*mask.size()[1])
        size_loss = self.coeffs["size"] * (torch.sum(mask))

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = (
            torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        )
        feat_size_loss = self.coeffs["feat_size"] * torch.mean(feat_mask)

        # entropy
        # This loss is designed to keep the mask as separated (either dense or
        # sparse as possible
        # if mask element ~ 1 i.e, 0.99: loss = ~0 - ~0 = 0
        # if mask element ~0 i.e, 0.01: loss = ~0 - 0 = 0
        # else, if mask element 0.5, loss = -log(0.5) = log(2) (maximum value)
        '''
        node_mask = self.get_node_mask()
        node_mask_ent = -node_mask*torch.log(node_mask) - (1 - node_mask) * torch.log(1- node_mask)
        '''

        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * (torch.mean(mask_ent))#  + torch.mean(node_mask_ent))


        # The same for feat mask entropy
        feat_mask_ent = - feat_mask             \
                        * torch.log(feat_mask)  \
                        - (1 - feat_mask)       \
                        * torch.log(1 - feat_mask)

        feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(feat_mask_ent)

        # laplacian
        list_D = []
        # print(self.masked_adj.size())
        for j in range(list(self.masked_adj.size())[-1]):
            list_D.append(torch.diag(
                            torch.sum(self.masked_adj[:, :, j], 0)
                            )
            )
        D = torch.stack(list_D, -1)
        # print(D.size(), self.masked_adj.size())
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        lap_loss = 0
        # TODO:
        '''
        # pred_label_t = torch.tensor(pred_label, dtype=torch.float)

        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            # @ is the matrix multiplication
            lap_loss = (self.coeffs["lap"]
                * (pred_label_t @ L @ pred_label_t)
                / self.adj.numel()
            )
        '''
        # grad
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss

    def log_mask(self, epoch):
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 3), dpi=400)
        plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/mask", tensorboardX.utils.figure_to_image(fig), epoch
        )

        # fig = plt.figure(figsize=(4,3), dpi=400)
        # plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")

        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)
        io_utils.log_matrix(
            self.writer, torch.sigmoid(self.feat_mask), "mask/feat_mask", epoch
        )

        fig = plt.figure(figsize=(4, 3), dpi=400)
        # use [0] to remove the batch dim
        plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        )

        if self.args.mask_bias:
            fig = plt.figure(figsize=(4, 3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.writer.add_image(
                "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
            )

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False

        if self.graph_mode:
            predicted_label = pred_label
            # adj_grad, x_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[0]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[0]
            x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        else:
            predicted_label = pred_label[node_idx]
            # adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[self.graph_idx]
            x_grad = x_grad[self.graph_idx][node_idx][:, np.newaxis]
            # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.adj).squeeze()
        if log_adj:
            io_utils.log_matrix(self.writer, adj_grad, "grad/adj_masked", epoch)
            self.adj.requires_grad = False
            io_utils.log_matrix(self.writer, self.adj.squeeze(), "grad/adj_orig", epoch)

        masked_adj = self.masked_adj[0].cpu().detach().numpy()

        # only for graph mode since many node neighborhoods for syn tasks are relatively large for
        # visualization
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, feat=self.x[0], threshold=None, max_component=False
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph_orig",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        io_utils.log_matrix(self.writer, x_grad, "grad/feat", epoch)

        adj_grad = adj_grad.detach().numpy()
        if self.graph_mode:
            print("GRAPH model")
            G = io_utils.denoise_graph(
                adj_grad,
                node_idx,
                feat=self.x[0],
                threshold=0.0003,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        else:
            # G = io_utils.denoise_graph(adj_grad, node_idx, label=label, threshold=0.5)
            G = io_utils.denoise_graph(adj_grad, node_idx, threshold_num=12)
            io_utils.log_graph(
                self.writer, G, name="grad/graph", epoch=epoch, args=self.args
            )

        # if graph attention, also visualize att

    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=self.x[0],
                threshold=0.2,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        else:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, threshold_num=12, max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
            )

