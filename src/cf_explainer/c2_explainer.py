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


class C2Explainer(ExplainerAlgorithm):
    r"""
    How to customize masks:
    - Edge masks: Given all_edge_index (edges to add, edges to delete), user_defined_hard_mask (which node can be added/deleted)
        num_hops, excluded_node_set, excluded_edge_set
    - Feature masks: node to be explained, global mask, normal mask
    - Node mask: excluded_node_set
        node mask by mask out node feature affect mean, sum values because edges still exist
        implement node_mask_to_edge_mask(self, egde_index)
    """
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
                 AR_mode = False,
                 FPM = False,
                 subgraph_mode = False,
                 silent_mode = False,
                 wo_ST = False,
                 AT_loss = False,
                 ENT_loss = False,
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

        self.AR_mode = AR_mode
        self.FPM = FPM
        self.subgraph_mode = subgraph_mode
        
        # for ablation study
        self.wo_ST = wo_ST
        self.AT_loss = AT_loss
        self.ENT_loss = ENT_loss
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
        
        if self.AT_loss:
            self.pred_same=True
        if index is not None: # Node classification task
            index = int(index)  # ARExplainer can only explain 1 node at once

            # extect the k-hop neighbor nodes around the node to be explained  
            subset, _, _, self.hard_edge_mask = k_hop_subgraph(
                index,
                num_hops=ExplainerAlgorithm._num_hops(model),
                edge_index=edge_index,
                relabel_nodes=False,
                num_nodes=x.size(0),
                flow=ExplainerAlgorithm._flow(model))
        
            total_edges = self.hard_edge_mask.sum()
        
            if not self.AR_mode:
                edge_index_delete = edge_index[:, self.hard_edge_mask]
                edge_index_remain = self._remove_edges(edge_index, edge_index_delete)

                # k-hop dense edge index (complete graph)
                dense_edge_index = torch.combinations(subset).T
                edge_index_add = dense_edge_index
                edge_index_add = to_undirected(edge_index_add).to(x.device)

                # delete existing edges from dense_edge_index
                edge_index_add = self._remove_edges(edge_index_add, edge_index)
            else:
                edge_index_add = torch.cat((torch.full(subset.shape, index).unsqueeze(-1).to(subset.device),subset.unsqueeze(-1)), dim=1).T

                subset, _, _, self.hard_edge_mask = k_hop_subgraph(
                        index,
                        num_hops=1,
                        edge_index=edge_index,
                        relabel_nodes=False,
                        num_nodes=x.size(0),
                        flow=ExplainerAlgorithm._flow(model))

                edge_index_delete = torch.cat((torch.full(subset.shape, index).unsqueeze(-1).to(subset.device),subset.unsqueeze(-1)), dim=1).T

                edge_index_delete = to_undirected(edge_index_delete).to(x.device)
                edge_index_delete = self._remove_edges(edge_index_delete, edge_index)
                edge_index_remain = self._remove_edges(edge_index, edge_index_delete)
                edge_index_add = to_undirected(edge_index_add).to(x.device)
                edge_index_add = self._remove_edges(edge_index_add, edge_index)
            if self.subgraph_mode:
                edge_index_add = torch.ones(size=[2,0]).to(x.device).to(int)
        else: # Graph classification task
            total_edges = edge_index.size(1)
            edge_index_remain = torch.ones(size=[2,0]).to(x.device).to(int) #empty tensor
            edge_index_delete = edge_index
            
            #======
            a = torch.arange(x.size(0))
            b = torch.arange(x.size(0), x.size(0)+3) # add three isolated nodes
            x = torch.cat([x, x[:3]]) # duplicate first three nodes
            edge_index_add = torch.cartesian_prod(a, b).T.to(x.device).to(int) # message flow?
            #======
            
            if self.subgraph_mode:
                edge_index_add = torch.ones(size=[2,0]).to(x.device).to(int)
            

        # Option: sampling from dense_edge_index
        # dense_edge_index = sampling(dense_edge_index)
        
        if self.undirected:
            edge_index_delete = self._sort_edge_index(edge_index_delete)
            edge_index_add = self._sort_edge_index(edge_index_add)
        
        all_edge_index = torch.cat([edge_index_remain, edge_index_delete, edge_index_add], dim=1)

        # orig_edge_mask, random init edge_mask
        self._initialize_masks(x, edge_index_remain, edge_index_delete, edge_index_add)

        # implement feature mask (three types)
        # the node to be explained: 1xd,
        # global node features: 1xd
        # node features: Nxd,

        # implement node mask: Nx1

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
            all_edge_index,
            target=target, index=index, **kwargs)

        best_cf, nearest_cf_label, min_perturbs, prop_perturbs = self._post_process_cfs(
            cfs, cf_labels, cf_perturbs, x, edge_index, total_edges)

        return Explanation(cf=best_cf, label=nearest_cf_label, perturbs=min_perturbs, prop_perturbs=prop_perturbs)

    def _train(self,
               model: torch.nn.Module,
               x: Tensor,
               all_edge_index: Tensor,
               *,
               target: Tensor,
               index=None,
               **kwargs):
        
        if not self.subgraph_mode:
            parameters = [self.edge_mask_delete, self.edge_mask_add]
        else:
            parameters = [self.edge_mask_delete]
            
        if self.FPM:
            parameters.append(self.feature_perturbation_matrix)
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        all_edge_mask = self._to_all_edge_mask()

        orig_mask = torch.cat([self.orig_mask_remain,
                               self.orig_mask_delete,
                               self.orig_mask_add])

        cfs = []
        cf_labels = []
        cf_perturbs = []
        for i in range(1, self.epochs+1):

            optimizer.zero_grad()
            
            if not self.FPM:
                h = x
            else:
                a = torch.zeros(x.shape).to(x.device)
                a[index,0] += 10*self.feature_perturbation_matrix
                h = x + a
                
            y_hat, y = model(h, edge_index=all_edge_index,
                             edge_weight=all_edge_mask, **kwargs), target
            if index is not None:
                y_hat = y_hat[index]
            else:
                y_hat = y_hat[0]
                
            if not self.AT_loss:
                cf_loss = self._cf_loss(y_hat)
            else:
                cf_loss = self._at_cf_loss(y_hat, y)
            
            perturb_loss = self._perturb_loss()
            
            # ent_loss = self._ent_loss()
            
            if self.ENT_loss:
                ent_loss = self._ent_loss()
                loss = cf_loss + self.coeffs['beta'] * perturb_loss + self.coeffs['gamma']*ent_loss
            else:
                loss = cf_loss + self.coeffs['beta'] * perturb_loss

            if self.print_loss:
                if self.ENT_loss:
                    print(
                    f"loss:{loss:.4f}, cf_loss:{cf_loss:.4f}, perturb_loss:{perturb_loss:.4f}, ent_loss:{ent_loss:.4f}")
                else:
                    print(
                    f"loss:{loss:.4f}, cf_loss:{cf_loss:.4f}, perturb_loss:{perturb_loss:.4f}")

            loss.backward()
            optimizer.step()

            all_edge_mask = self._to_all_edge_mask()

            with torch.no_grad():
                if self.wo_ST:
                    cf_edge_mask = self._binarilize_mask(all_edge_mask.detach().clone(), 0.5)
                else:
                    cf_edge_mask=all_edge_mask.detach().clone()
                cf_edge_index = all_edge_index[:, cf_edge_mask.to(torch.bool)]
                assert is_undirected(cf_edge_index)
                y_hat = model(h, cf_edge_index, **kwargs)
                if index is not None:
                    y_hat = y_hat[index]
                else:
                    y_hat = y_hat[0]
                if torch.argmax(y_hat).item() != y:
                    perturb_mat = (cf_edge_mask - orig_mask).detach().clone()
                    num_perturb = torch.sum(torch.abs(perturb_mat)).item()
                    if self.FPM:
                        num_perturb += 2*torch.abs(self.feature_perturbation_matrix)
                    cfs.append(cf_edge_index)
                    cf_perturbs.append(num_perturb)
                    cf_labels.append(torch.argmax(y_hat).item())
                if self.AT_loss:
                    if torch.argmax(y_hat).item() != y:
                        self.pred_same = False
                    else:
                        self.pred_same = True

        return cfs, cf_labels, cf_perturbs

    def _initialize_masks(self,
                          x: Tensor,
                          edge_index_remain: Tensor,
                          edge_index_delete: Tensor,
                          edge_index_add: Tensor):
        
        (N, F) = x.size()
        E_remain = edge_index_remain.size(1)
        E_delete, E_add = edge_index_delete.size(1), edge_index_add.size(1)
        
        if self.print_loss:
            print(E_remain, E_delete, E_add)
        
        self.orig_mask_remain = torch.ones(E_remain).to(x.device)
        self.orig_mask_delete = torch.ones(E_delete).to(x.device)
        self.orig_mask_add = torch.zeros(E_add).to(x.device)

        # random init
        edge_mask_delete = torch.rand(E_delete).to(x.device)*2-1
        edge_mask_add = torch.rand(E_add).to(x.device)*2-1
        
        # masks to optimize
        self.edge_mask_delete = Parameter(edge_mask_delete)
        if not self.subgraph_mode:
            self.edge_mask_add = Parameter(edge_mask_add)
        else:
            self.edge_mask_add = edge_mask_add
            
        if self.FPM:
            feature_perturbation_matrix = torch.tensor(0.).to(x.device)
            self.feature_perturbation_matrix = Parameter(feature_perturbation_matrix)

    def _perturb_loss(self) -> Tensor:
        a = torch.sigmoid(self.edge_mask_delete) - self.orig_mask_delete
        b = torch.sigmoid(self.edge_mask_add) - self.orig_mask_add
        perturb_loss = torch.sum(torch.abs(a)) + torch.sum(torch.abs(b))
        
        if self.FPM:
            perturb_loss += torch.abs(self.feature_perturbation_matrix)
        
        return perturb_loss

    def _ent_loss(self) -> Tensor:
        a = torch.cat([self.edge_mask_delete, self.edge_mask_add])
        m = torch.sigmoid(a)
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
            
    def _at_cf_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        y = torch.tensor(y).to(y_hat.device)
        cf_loss= -F.cross_entropy(y_hat, y)
        return self.pred_same*cf_loss

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
        self.edge_mask_add = None
        self.edge_mask_delete = None
        self.orig_mask_remain = None
        self.orig_mask_add = None
        self.orig_mask_delete = None
        if self.AT_loss:
            self.pred_same=None

        self.hard_edge_mask = None
        self.desired_y = None

    @staticmethod
    def _sort_edge_index(edge_index):
        edge_index = sort_edge_index(edge_index)
        row, col = edge_index[0], edge_index[1]
        edge_index_triu = edge_index[:, row < col]
        edge_index_tril = edge_index_triu.new_zeros(edge_index_triu.size())
        edge_index_tril[0], edge_index_tril[1] = edge_index_triu[1], edge_index_triu[0]
        edge_index_tril = edge_index_tril.flip(dims=[1])
        edge_index = torch.cat((edge_index_triu, edge_index_tril), dim=1)
        return edge_index

    def _to_all_edge_mask(self):
        edge_mask_delete = torch.sigmoid(self.edge_mask_delete)
        edge_mask_add = torch.sigmoid(self.edge_mask_add)

        def _to_undirected_mask(edge_mask):  # without slicing, very important!
            edge_mask = edge_mask + edge_mask.flip(dims=[0])
            edge_mask = edge_mask/2
            return edge_mask
        
        if self.undirected:
            edge_mask_delete = _to_undirected_mask(edge_mask_delete)
            edge_mask_add = _to_undirected_mask(edge_mask_add)
        
        if self.wo_ST:
            all_edge_mask = torch.cat([self.orig_mask_remain,
                                   edge_mask_delete,
                                   edge_mask_add])
        else:
            all_edge_mask = torch.cat([self.orig_mask_remain,
                                   self._ST_trick(edge_mask_delete),
                                   self._ST_trick(edge_mask_add)])
        return all_edge_mask

    @staticmethod
    def _ST_trick(edge_mask):
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
