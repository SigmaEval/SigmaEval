"""
Custom exceptions for SigmaEval.
"""


class LLMCommunicationError(Exception):
    """
    Raised when there is any failure communicating with an LLM or parsing its
    response in a way required by SigmaEval's protocols.

    This includes transport/API errors, timeouts, invalid/malformed outputs, or
    missing required fields from responses when a specific schema is expected.
    """

    pass


