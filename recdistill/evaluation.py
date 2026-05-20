from __future__ import annotations
import math
import torch


def evaluate_embeddings(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    train_seen: dict[int, set[int]],
    ground_truth: dict[int, set[int]],
    top_k: int,
    batch_size: int,
    device: torch.device,
    scorer=None,
) -> tuple[dict[str, float], int]:
    eval_users = sorted(user for user, items in ground_truth.items() if items)
    if not eval_users:
        return _metrics_at_k({}, ground_truth, top_k), 0

    recommendations, leaked_users = _build_topk_recommendations(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        users=eval_users,
        train_seen=train_seen,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
        scorer=scorer,
    )
    return _metrics_at_k(recommendations, ground_truth, top_k), leaked_users


def evaluate_student(
    model: torch.nn.Module,
    train_seen: dict[int, set[int]],
    val_gt: dict[int, set[int]],
    test_gt: dict[int, set[int]],
    top_k: int,
    batch_size: int,
    device: torch.device,
    eval_val_only: bool = False,
) -> dict[str, dict[str, float] | int]:
    model.eval()
    user_embeddings = model.get_all_user_embeddings().detach()
    item_embeddings = model.get_all_item_embeddings().detach()
    scorer = model if hasattr(model, "score_items_for_user") else None
    val_metrics, val_leaks = evaluate_embeddings(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        train_seen=train_seen,
        ground_truth=val_gt,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
        scorer=scorer,
    )
    if eval_val_only:
        return {"val": val_metrics, "leaked_users_val": val_leaks}
    test_metrics, test_leaks = evaluate_embeddings(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        train_seen=train_seen,
        ground_truth=test_gt,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
        scorer=scorer,
    )
    return {
        "val": val_metrics,
        "test": test_metrics,
        "leaked_users_val": val_leaks,
        "leaked_users_test": test_leaks,
    }


def _build_topk_recommendations(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    users: list[int],
    train_seen: dict[int, set[int]],
    top_k: int,
    batch_size: int,
    device: torch.device,
    scorer=None,
) -> tuple[dict[int, list[int]], int]:
    if item_embeddings is None and scorer is None:
        raise ValueError("Evaluation requires item embeddings or a scorer.")
    num_items = int(item_embeddings.size(0)) if item_embeddings is not None else int(scorer.num_items)
    k = max(1, min(int(top_k), num_items))
    user_emb = user_embeddings.to(device) if user_embeddings is not None else None
    item_emb = item_embeddings.to(device) if item_embeddings is not None else None
    recommendations: dict[int, list[int]] = {}
    leaked_users = 0

    with torch.no_grad():
        if scorer is not None:
            scorer = scorer.to(device)
            for user in users:
                scores = scorer.score_items_for_user(user=int(user), num_items=num_items).detach().to(device)
                seen = train_seen.get(user, set())
                if seen:
                    scores[torch.tensor(sorted(seen), dtype=torch.long, device=device)] = float("-inf")
                top_items = torch.topk(scores, k=k, dim=0).indices.detach().cpu().tolist()
                raw_topk = [int(item) for item in top_items[:top_k]]
                if seen and any(item in seen for item in raw_topk):
                    leaked_users += 1
                recommendations[user] = [item for item in raw_topk if item not in seen][:top_k]
            return recommendations, leaked_users

        for offset in range(0, len(users), batch_size):
            batch_users = users[offset : offset + batch_size]
            if user_emb is None or item_emb is None:
                raise ValueError("Embedding evaluation path requires user and item embeddings.")
            user_idx = torch.tensor(batch_users, dtype=torch.long, device=device)
            scores = torch.matmul(user_emb[user_idx], item_emb.transpose(0, 1))
            for row_idx, user in enumerate(batch_users):
                seen = train_seen.get(user, set())
                if seen:
                    scores[row_idx, torch.tensor(sorted(seen), dtype=torch.long, device=device)] = float("-inf")
            top_items = torch.topk(scores, k=k, dim=1).indices.detach().cpu().tolist()
            for row_idx, user in enumerate(batch_users):
                seen = train_seen.get(user, set())
                raw_topk = [int(item) for item in top_items[row_idx][:top_k]]
                if seen and any(item in seen for item in raw_topk):
                    leaked_users += 1
                recommendations[user] = [item for item in raw_topk if item not in seen][:top_k]
    return recommendations, leaked_users


def _metrics_at_k(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    top_k: int,
) -> dict[str, float]:
    users = [user for user, items in ground_truth.items() if items]
    if not users:
        return {"users": 0, "precision": 0.0, "recall": 0.0, "ndcg": 0.0, "hr": 0.0}

    precision_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0
    hr_sum = 0.0
    for user in users:
        recs = recommendations.get(user, [])[:top_k]
        gt = ground_truth[user]
        hits = [1 if item in gt else 0 for item in recs]
        num_hits = sum(hits)
        precision_sum += num_hits / max(1, top_k)
        recall_sum += num_hits / max(1, len(gt))
        hr_sum += 1.0 if num_hits > 0 else 0.0
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_len = min(len(gt), top_k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_len))
        ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

    total = float(len(users))
    return {
        "users": int(total),
        "precision": precision_sum / total,
        "recall": recall_sum / total,
        "ndcg": ndcg_sum / total,
        "hr": hr_sum / total,
    }
