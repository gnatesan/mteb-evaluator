from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

#Function to calculate the energy of all vectors in a query
def ed_calc(x):
    M = x.shape[0]
    x_expanded = x.unsqueeze(1).expand(-1, M, -1)
    ed_sum = torch.norm(x_expanded - x, dim=2).sum()
    return ed_sum / (M * M)


#Function to calcjulate the energy between a single vecctor document and multi-vector query
def energy_calc(x, y):
    M = x.shape[0]
    N = y.shape[0]

    # Expand tensors to create all pairs of vectors
    x_expanded = x.unsqueeze(1).expand(-1, N, -1)
    y_expanded = y.unsqueeze(0).expand(M, -1, -1)
    
    # Compute pairwise squared Euclidean distances
    pairwise_diff = x_expanded - y_expanded
    squared_distances = torch.sum(pairwise_diff ** 2, dim=2)

    # Sum up squared distances and scale by (M * N)
    ed_sum = torch.sum(torch.sqrt(squared_distances))

    return 2 * ed_sum / (M * N)


#Function to calculate the energy distance between a batch of queries and a batch of documents.
#Queries are represented as a list of 2d tensors, and documents are represented by a list of 
#1d tensors. Implementation is based off of https://pages.stat.wisc.edu/~wahba/stat860public/pdf4/Energy/EnergyDistance10.1002-wics.1375.pdf
def energy_distance(x, y):

    num_queries = len(x) #number of queries
    num_documents = len(y) #number of documents

    #print("Num queries:", num_queries)
    #print("Num documents:", num_documents)

    # Pre-calculate energy for all queries
    ed_queries = torch.stack([ed_calc(query) for query in x])

    # Create a tensor of shape M*N filled with zeros
    tensor = torch.zeros(num_queries, num_documents)

    for i in range(num_queries):
      ed_query = ed_queries[i] #store energy calculation of query to improve runtime
      for j in range(num_documents):
        tensor[i][j] = energy_calc(x[i], y[j].reshape(1,-1)).item() - ed_query.item()

    return tensor
	

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


# From https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/custom_metrics.py#L4
def mrr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), 5)
        logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR


def recall_cap(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    capped_recall = {}

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = 0.0

    k_max = max(k_values)
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        query_relevant_docs = [
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        ]
        for k in k_values:
            retrieved_docs = [
                row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0
            ]
            denominator = min(len(query_relevant_docs), k)
            capped_recall[f"R_cap@{k}"] += len(retrieved_docs) / denominator

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = round(capped_recall[f"R_cap@{k}"] / len(qrels), 5)
        logging.info("R_cap@{}: {:.4f}".format(k, capped_recall[f"R_cap@{k}"]))

    return capped_recall


def hole(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    Hole = {}

    for k in k_values:
        Hole[f"Hole@{k}"] = 0.0

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)
    logging.info("\n")

    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        for k in k_values:
            hole_docs = [
                row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus
            ]
            Hole[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"] / len(qrels), 5)
        logging.info("Hole@{}: {:.4f}".format(k, Hole[f"Hole@{k}"]))

    return Hole


def top_k_accuracy(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    top_k_acc = {}

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0]
            for item in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:k_max]
        ]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"] / len(qrels), 5)
        logging.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    return top_k_acc
