"""
Defines and generates writing style variations for the user simulator.
"""

import random
from typing import Dict, Optional
from .models import WritingStyleAxes


def _generate_writing_style(
    axes: Optional[WritingStyleAxes] = None,
) -> Dict[str, str]:
    """
    Generate a random writing style from the available axes.
    """
    if axes is None:
        axes = WritingStyleAxes()

    style = {
        "Proficiency": random.choice(axes.proficiency),
        "Tone": random.choice(axes.tone),
        "Verbosity": random.choice(axes.verbosity),
        "Formality": random.choice(axes.formality),
    }

    return style
