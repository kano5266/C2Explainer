# C2Explainer: FAccT 2025

**C2Explainer: Customizable Mask-based Counterfactual Explanation for
Graph Neural Networks**\
Jiali Ma, Ichigaku Takigawa, Akihiro Yamamoto\
ACM Conference on Fairness, Accountability, and Transparency (FAccT),
2025

## Note
### This implementation is PyG based
This implementation fully supports the PyTorch Geometric sparse graph representation.

In particular, the graph structure is represented using `edge_index` instead of dense adjacency matrices (`adj_matrix`).
All graph operations are implemented using PyTorch Geometric’s sparse message passing framework.

### Edge mask in explanation
This implementation utilizes the explain mechanism in PyTorch Geometric.

During explanation, the learned `edge_mask` is injected as `edge_weight` inside the GCN convolution:

``` python
if self._explain:
    edge_weight = self._edge_mask
```
This allows the explanation mask to directly control the edge weights used during message passing.
