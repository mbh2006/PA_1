"""Task 4 Step 3: GCSC - the vanilla recipe plus RandAugment(2, 9).

Only the augmentation changes (inserted after crop/flip, before normalisation);
initialisation, optimiser, schedule, batch size, epochs, seed and the
checkpoint rule are exactly the vanilla ones.
"""
from __future__ import annotations

from task4.methods.vanilla import Vanilla


class GCSC(Vanilla):
    name = "gcsc"
