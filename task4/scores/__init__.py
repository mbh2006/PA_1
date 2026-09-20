"""Unknownness scores for Task 4.

All post-hoc scores operate on the SAME frozen model outputs: the ten known
logits (and, for Mahalanobis, the 512-d penultimate feature). Larger score
means "more unknown".
"""
