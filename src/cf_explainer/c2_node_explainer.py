from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.utils import sort_edge_index, k_hop_subgraph, coalesce
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel, ModelReturnType


# edge weight version, for node classification tasks


class C2NodeExplainer(ExplainerAlgorithm):
    coeffs = {
        'beta': 1,  # for perturb loss
        'gamma': 0.1,  # for entropy loss
        'EPS': 1e-15,  # epsilon
    }

    def __init__(self,
                 epochs,
                 lr,
                 *,
                 print_loss=False,
                 silent_mode = False,
                 desired_y = None,
                 undirected = True,
                 **kwargs
                 ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.undirected = True
        self.coeffs.update(kwargs)

        self.hard_edge_mask = None
        self.edge_mask_add = None
        self.edge_mask_delete = None
        self.orig_mask_remain = None
        self.orig_mask_add = None
        self.orig_mask_delete = None

        self.desired_y = desired_y
        self.undirected = undirected

        self.print_loss = print_loss
        self.silent_mode = silent_mode

    def forward(self,
                model: torch.nn.Module,
                x: Tensor,
                edge_index: Tensor,
                *,  # the end of the positional arguments.
                target: Tensor,
                index: Optional[int] = None,
                **kwargs
                ):
        
        total_edges = edge_index.size(1)

        # orig_edge_mask, random init edge_mask
        self._initialize_masks(x, edge_index)

        # explain model output, not the ground-truth label
        if index is not None:
            pred = model(x, edge_index, **kwargs)[index].detach()
        else:
            pred = model(x, edge_index, **kwargs)[0].detach()
        target = torch.argmax(pred).item()  # get orig model prediciton

        # get the desired_y if not given
        if self.desired_y is None:
            _, indices = pred.topk(2)
            self.desired_y = indices[1]

        cfs, cf_labels, cf_perturbs = self._train(
            model, x,
            edge_index,
            target=target, index=index, **kwargs)

        best_cf, nearest_cf_label, min_perturbs, prop_perturbs = self._post_process_cfs(
            cfs, cf_labels, cf_perturbs, x, edge_index, total_edges)

        return Explanation(cf=best_cf, label=nearest_cf_label, perturbs=min_perturbs, prop_perturbs=prop_perturbs)

    def _train(self,
               model: torch.nn.Module,
               x: Tensor,
               edge_index: Tensor,
               *,
               target: Tensor,
               index=None,
               **kwargs):
        
        parameters = [self.node_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        edge_mask = self._to_edge_mask(edge_index)

        orig_mask = self.orig_mask

        cfs = []
        cf_labels = []
        cf_perturbs = []
        for i in range(1, self.epochs+1):

            optimizer.zero_grad()

            y_hat, y = model(x, edge_index=edge_index,
                             edge_weight=edge_mask, **kwargs), target
            if index is not None:
                y_hat = y_hat[index]
            else:
                y_hat = y_hat[0]
                
            cf_loss = self._cf_loss(y_hat)
            
            perturb_loss = self._perturb_loss(edge_mask)
            
            ent_loss = self._ent_loss()

            loss = cf_loss + self.coeffs['beta'] * perturb_loss # + self.coeffs['gamma']*ent_loss

            if self.print_loss:
                print(
                    f"loss:{loss:.4f}, cf_loss:{cf_loss:.4f}, perturb_loss:{perturb_loss:.4f}")

            loss.backward()
            optimizer.step()

            edge_mask = self._to_edge_mask(edge_index)

            with torch.no_grad():
                cf_edge_index = edge_index[:, edge_mask.to(torch.bool)]
                assert is_undirected(cf_edge_index)
                y_hat = model(x, cf_edge_index, **kwargs)
                if index is not None:
                    y_hat = y_hat[index]
                else:
                    y_hat = y_hat[0]
                if torch.argmax(y_hat).item() != y:
                    perturb_mat = (edge_mask - orig_mask).detach().clone()
                    num_perturb = torch.sum(torch.abs(perturb_mat)).item()
                    cfs.append(cf_edge_index)
                    cf_perturbs.append(num_perturb)
                    cf_labels.append(torch.argmax(y_hat).item())

        return cfs, cf_labels, cf_perturbs

    def _initialize_masks(self,
                          x: Tensor,
                          edge_index: Tensor):
        
        (N, F) = x.size() # x
        E = edge_index.size(1)
        
        # node_mask = torch.rand(N,1).to(x.device)
        # node_mask = node_mask*2-1
        node_mask = torch.ones(N,1).to(x.device)
        self.node_mask = Parameter(node_mask)
        
        self.orig_mask = torch.ones(E).to(x.device)
            
    def _to_edge_mask(self, edge_index): 
        # threshold is 1 not 0.5
        # to prevent 0.7 * 0.7 = 0.49 < 0.5
        node_mask = torch.sigmoid(self.node_mask) 
        node_mask = self._ST_trick(node_mask)
        dense_edge_mask = torch.mm(node_mask, node_mask.T)
        # dense_edge_mask = torch.sigmoid(dense_edge_mask) 
        # sym_mask = (sym_mask + sym_mask.t()) / 2 # symmetric
        edge_mask = dense_edge_mask[tuple(edge_index)] 
        # edge_mask = self._ST_trick(edge_mask)
        
        return edge_mask

    def _perturb_loss(self, edge_mask) -> Tensor:
        a = torch.sigmoid(edge_mask) - self.orig_mask
        perturb_loss = torch.sum(torch.abs(a))
        return perturb_loss

    def _ent_loss(self) -> Tensor:
        m = torch.sigmoid(self.node_mask)
        ent_loss = (-m * torch.log(m + self.coeffs['EPS']) - (
            1 - m) * torch.log(1 - m + self.coeffs['EPS'])).sum()
        return ent_loss

    def _cf_loss(self, y_hat: Tensor) -> Tensor:
        y_vec = y_hat.new_zeros(y_hat.size())
        y_vec[self.desired_y] = 1.0

        # print("y_hat,y_vec",y_hat,y_vec)

        if self.model_config.return_type == ModelReturnType.raw:
            return F.cross_entropy(y_hat, y_vec)
        elif self.model_config.return_type == ModelReturnType.probs:
            y_hat = y_hat.log()
            return F.nll_loss(y_hat, y_vec)
        elif self.model_config.return_type == ModelReturnType.log_probs:
            return F.nll_loss(y_hat, y_vec)
        else:
            assert False

    def _post_process_cfs(self, cfs, cf_labels, cf_perturbs, x, edge_index, total_edges):
        if not cfs:
            best_cf = None
            min_perturbs = None
            prop_perturbs = None
            nearest_cf_label = None
        else:
            min_perturbs = min(cf_perturbs)
            min_perturb_index = cf_perturbs.index(min_perturbs)
            prop_perturbs = min_perturbs / total_edges

            best_cf = cfs[min_perturb_index]

            nearest_cf_label = cf_labels[min_perturb_index]
            if not self.silent_mode:
                print(
                    f"num_cfs:{len(cf_perturbs)}, min_perturbs:{min_perturbs}, prop_perturbs:{prop_perturbs}, cf_label {nearest_cf_label}")
                print("=========")

        self._clean_model()
        return best_cf, nearest_cf_label, min_perturbs, prop_perturbs

    def _clean_model(self):
        self.node_mask = None
        self.orig_mask_remain = None
        self.orig_mask_delete = None

        self.hard_edge_mask = None
        self.desired_y = None

    def _ST_trick(self, edge_mask):
        binarized_edge_mask = (edge_mask > 0.5).detach().clone().to(torch.int)
        edge_mask = binarized_edge_mask - edge_mask.detach() + edge_mask
        return edge_mask
    
    @staticmethod
    def _binarilize_mask(mask, binarilize_threshold):
        mask[mask >= binarilize_threshold] = 1
        mask[mask < binarilize_threshold] = 0
        return mask

    @staticmethod
    def _remove_edges(edge_index, edge_index_to_remove):
        r"""
        remove edges in edge_index that are also in edge_index_to_remove.
        """
        # Trick from https://github.com/pyg-team/pytorch_geometric/discussions/9440
        all_edge_index = torch.cat([edge_index,
                                    edge_index_to_remove], dim=1)

        # mark removed edges as 1 and 0 otherwise
        all_edge_weights = torch.cat([torch.zeros(edge_index.size(1)),
                                      torch.ones(edge_index_to_remove.size(1))]
                                     ).to(all_edge_index.device)

        all_edge_index, all_edge_weights = coalesce(
            all_edge_index, all_edge_weights)

        # remove edges indicated by 1
        edge_index = all_edge_index[:, all_edge_weights == 0]
        return edge_index

    def supports(self) -> bool:
        return True

