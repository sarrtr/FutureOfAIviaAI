# GraphSAGE + pair-MLP approach for forecasting the future of AI (link prediction)

## Methodology

The method implements hybrid graph-based model which consists of:
1. **GraphSAGE Encoder:** 
- Learns node embeddings from graph structure.
2. **Feature Engineering:** 
- Embedding-based features:
	- Concatenation of u,v embeddings
	- Element-wise product
	- Absolute difference
- Structural graph features:
	- Common neighbors count
	- Jaccard coefficient
	- Adamic-Adar index
	- Node degrees
	- Shortest path length (capped)
3. **MLP Classifier:**
- Perform binary classification of combined feature vectors.

This method combines learned embeddings with handcrafted graph features which corresponds to paper's best-performing approach (M1) and leverages GraphSAGE which handle large graphs via neighborhood sampling. In comparison with other GNNs, GraphSAGE can perform inductive learning (handle new nodes in the graph) and it consumes less memory.

However, two-stage training (GraphSAGE + MLP) is still expensive, and  random features may not capture all semantics.

The method is relevant to the task because it combines ML with network features and can handle large-scale data.

## Evaluation

Evaluation metrics:
- **AUC-ROC**: ranking quality between positives and negatives.
- **Average Precision (AP)**: quality on imbalanced data.
- **Precision@K** (K=100): quality of top predictions.