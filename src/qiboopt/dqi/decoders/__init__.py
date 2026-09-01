"""
In-circuit syndrome decoders for Decoded Quantum Interferometry.

A :class:`SyndromeDecoder` produces a table of ``(y, S)`` pairs that the
DQI circuit consumes to uncompute the m-qubit error register conditional
on the n-qubit solution register holding ``y``. Decoding is therefore
performed by the quantum circuit itself; there is no classical
post-processing step on measurement outcomes.

Phase 1 ships only the brute-force lookup-table decoder
(:class:`LUTDecoder`).
"""

from qiboopt.dqi.decoders.base import SyndromeDecoder
from qiboopt.dqi.decoders.lut import LUTDecoder

DECODER_REGISTRY = {
    "lut": LUTDecoder,
}


def get_decoder(name, *args, **kwargs):
    """Instantiate a registered decoder by name.

    Args:
        name (str): Decoder identifier; currently one of ``"lut"``.
        args: Forwarded positionally to the decoder constructor.
        kwargs: Forwarded as keywords to the decoder constructor.

    Returns:
        :class:`SyndromeDecoder`: Instantiated decoder.
    """
    if name not in DECODER_REGISTRY:
        from qibo.config import raise_error

        raise_error(
            KeyError,
            f"Unknown decoder {name!r}; available: {sorted(DECODER_REGISTRY)}.",
        )
    return DECODER_REGISTRY[name](*args, **kwargs)


__all__ = ["SyndromeDecoder", "LUTDecoder", "DECODER_REGISTRY", "get_decoder"]
