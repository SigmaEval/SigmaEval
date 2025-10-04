"""
Defines and generates writing style variations for the user simulator.
"""

import secrets

from .models import WritingStyleAxes


def _generate_writing_style(axes: WritingStyleAxes) -> str:
    """
    Generates a single writing style instruction sentence by randomly combining axes.
    
    Internal implementation detail - API may change without backward compatibility.
    
    Args:
        axes: A WritingStyleAxes object containing the values for each axis.
    """
    proficiency = secrets.choice(axes.proficiency)
    tone = secrets.choice(axes.tone)
    verbosity = secrets.choice(axes.verbosity)
    formality = secrets.choice(axes.formality)

    return (
        "- Adopt the following writing style for the user:\n"
        f"    - Proficiency: {proficiency}\n"
        f"    - Tone: {tone}\n"
        f"    - Verbosity: {verbosity}\n"
        f"    - Formality: {formality}\n\n"
        "    (Note: If any aspect of this writing style conflicts with the 'Given' "
        "(background) or 'When' (scenario) instructions noted above, you "
        "must prioritize those instructions and disregard the conflicting "
        "aspects of this writing style.)"
    )
