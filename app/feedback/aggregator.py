import logging
from collections import Counter, defaultdict

import numpy as np

from app.models.schemas import Annotation

logger = logging.getLogger(__name__)


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Compute Cohen's Kappa for two annotators."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    p_observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    all_labels = list(set(labels_a + labels_b))
    p_expected = sum(
        (labels_a.count(l) / n) * (labels_b.count(l) / n)
        for l in all_labels
    )
    if p_expected == 1.0:
        return 1.0
    return (p_observed - p_expected) / (1 - p_expected)


def _raw_agreement(labels: list[str]) -> float:
    if not labels:
        return 0.0
    most_common_count = Counter(labels).most_common(1)[0][1]
    return most_common_count / len(labels)


class AnnotationAggregator:
    def compute_agreement(self, annotations: list[Annotation]) -> dict:
        if not annotations:
            return {
                "by_type": {},
                "overall_agreement": 1.0,
                "overall_kappa": 1.0,
                "disagreements": [],
            }

        by_type: dict[str, list[Annotation]] = defaultdict(list)
        for ann in annotations:
            by_type[ann.type].append(ann)

        type_results: dict[str, dict] = {}
        disagreements: list[dict] = []
        all_agreements: list[float] = []
        all_kappas: list[float] = []

        for ann_type, anns in by_type.items():
            labels = [a.label for a in anns]
            label_counts = dict(Counter(labels))
            agreement = _raw_agreement(labels)
            all_agreements.append(agreement)

            # Compute kappa
            if len(anns) == 2:
                kappa = _cohen_kappa([labels[0]], [labels[1]])
            elif len(anns) >= 3:
                # Average pairwise kappa (approximation of Fleiss')
                kappas: list[float] = []
                for i in range(len(anns)):
                    for j in range(i + 1, len(anns)):
                        kappas.append(_cohen_kappa([labels[i]], [labels[j]]))
                kappa = float(np.mean(kappas)) if kappas else 0.0
            else:
                kappa = 1.0

            all_kappas.append(kappa)
            type_results[ann_type] = {
                "agreement": round(agreement, 4),
                "kappa": round(kappa, 4),
                "labels": label_counts,
            }

            if agreement < 1.0:
                disagreements.append({
                    "type": ann_type,
                    "annotator_ids": [a.annotator_id for a in anns],
                    "labels": labels,
                })

        return {
            "by_type": type_results,
            "overall_agreement": round(float(np.mean(all_agreements)), 4) if all_agreements else 1.0,
            "overall_kappa": round(float(np.mean(all_kappas)), 4) if all_kappas else 1.0,
            "disagreements": disagreements,
        }

    def resolve_disagreements(self, annotations: list[Annotation], threshold: float) -> dict:
        if not annotations:
            return {"resolved": {}, "needs_review": []}

        by_type: dict[str, list[Annotation]] = defaultdict(list)
        for ann in annotations:
            by_type[ann.type].append(ann)

        resolved: dict[str, str] = {}
        needs_review: list[str] = []

        for ann_type, anns in by_type.items():
            labels = [a.label for a in anns]
            agreement = _raw_agreement(labels)
            if agreement >= threshold:
                resolved[ann_type] = Counter(labels).most_common(1)[0][0]
            else:
                needs_review.append(ann_type)

        return {"resolved": resolved, "needs_review": needs_review}
