import torch
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    generate_binary_structure,
)
from utils.pytuils import AverageMeter


class Metric:
    def __init__(self) -> None:
        self.dice = AverageMeter()
        self.jaccard = AverageMeter()
        self.assd = AverageMeter()
        self.hd95 = AverageMeter()

    def add(self, result: torch.Tensor, reference: torch.Tensor) -> None:
        self.dice.add(*dice(result, reference))
        self.jaccard.add(*jaccard(result, reference))
        self.assd.add(*assd(result, reference))
        self.hd95.add(*hd95(result, reference))

    def __str__(self) -> str:
        return f"dice:\t\t{self.dice.get()*100:.2f}±{self.dice.get_std()*100:.2f}\niou (jaccard):\t{self.jaccard.get()*100:.2f}±{self.jaccard.get_std()*100:.2f}\nassd:\t\t{self.assd.get():.2f}±{self.assd.get_std():.2f}\nhd95:\t\t{self.hd95.get():.2f}±{self.hd95.get_std():.2f}"

    def __repr__(self) -> str:
        return self.__str__()


def dice(
    result: torch.Tensor, reference: torch.Tensor, epsilon=1e-6
) -> tuple[list, int]:
    assert result.shape == reference.shape, f"{result.shape}, {reference.shape}"
    assert result.dim() == 4, result.shape

    intersection = torch.sum(result * reference, dim=(2, 3))
    sum_mask = torch.sum(result, dim=(2, 3))
    sum_target = torch.sum(reference, dim=(2, 3))
    dice_score = ((2 * intersection) + epsilon) / (sum_mask + sum_target + epsilon)

    # Flatten and remove NaNs - return list of individual scores
    valid_scores = dice_score[~torch.isnan(dice_score)].cpu().tolist()
    return valid_scores, len(valid_scores)


def jaccard(result: torch.Tensor, reference: torch.Tensor) -> tuple[list, int]:
    assert result.shape == reference.shape, f"{result.shape}, {reference.shape}"

    epsilon = 1e-6

    intersection = torch.sum(result * reference, dim=(2, 3))
    sum_mask = torch.sum(result, dim=(2, 3))
    sum_target = torch.sum(reference, dim=(2, 3))
    jaccard_score = ((intersection) + epsilon) / (
        sum_mask + sum_target - intersection + epsilon
    )

    # Flatten and remove NaNs - return list of individual scores
    valid_scores = jaccard_score[~torch.isnan(jaccard_score)].cpu().tolist()
    return valid_scores, len(valid_scores)


def surface_distances(
    result: np.ndarray, reference: np.ndarray, voxelspacing=None, connectivity=1
):
    result, reference = np.atleast_1d(result.astype(np.bool_)), np.atleast_1d(
        reference.astype(np.bool_)
    )
    footprint = generate_binary_structure(result.ndim, connectivity)

    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def assd(
    result: torch.Tensor, reference: torch.Tensor, voxelspacing=None, connectivity=1
):
    result, reference = result.cpu().numpy(), reference.cpu().numpy()

    assd_scores = []
    for b in range(len(result)):
        _result, _reference = result[b], reference[b]
        if np.sum(_result) != 0 and np.sum(_reference) != 0:
            sd1 = surface_distances(_result, _reference, voxelspacing, connectivity)
            sd2 = surface_distances(_reference, _result, voxelspacing, connectivity)
            if len(sd1) > 0 and len(sd2) > 0:
                score = (np.mean(sd1) + np.mean(sd2)) / 2.0
                assd_scores.append(score)
    return assd_scores, len(assd_scores)


def hd95(
    result: torch.Tensor, reference: torch.Tensor, voxelspacing=None, connectivity=1
):
    result, reference = result.cpu().numpy(), reference.cpu().numpy()
    hd95_scores = []
    for b in range(len(result)):
        _result, _reference = result[b], reference[b]
        if np.sum(_result) != 0 and np.sum(_reference) != 0:
            hd1 = surface_distances(_result, _reference, voxelspacing, connectivity)
            hd2 = surface_distances(_reference, _result, voxelspacing, connectivity)
            score = np.percentile(np.hstack((hd1, hd2)), 95)
            hd95_scores.append(score)
    return hd95_scores, len(hd95_scores)
