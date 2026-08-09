"""Legal LOB coordinate transform.

The transform maps unconstrained model coordinates to a valid LOB snapshot.
Both input and output use the project format [B, 2, 20, T].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


COORDINATE_REPRESENTATIONS = frozenset(
    {"matched_lob", "legal_lob", "legal_lob_ba"}
)


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=1e-12)
    return x + torch.log(-torch.expm1(-x))


@dataclass(frozen=True)
class LegalLOBTransform:
    """Encode/decode between raw parameters and valid LOB states.

    Price parameter rows:
      0       mid
      1       spread_raw
      2:11    ask gap raw values for levels 2..10
      11:20   bid gap raw values for levels 2..10

    Volume parameter rows:
      0:20    volume_raw
    """

    min_spread: float = 1e-6
    gap_eps: float = 1e-8
    volume_eps: float = 0.0

    def _decode_gap_excess(self, params: torch.Tensor) -> torch.Tensor:
        """Map unconstrained gap coordinates to strictly positive excess."""
        return F.softplus(params)

    def _encode_gap_excess(self, excess: torch.Tensor) -> torch.Tensor:
        """Encode strictly positive gap excess into unconstrained coordinates."""
        return _inverse_softplus(torch.clamp(excess, min=1e-12))

    def _guard_ask_level(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        positive_inf: torch.Tensor,
    ) -> torch.Tensor:
        floor = torch.nextafter(
            (previous + self.gap_eps).detach(),
            positive_inf,
        )
        return torch.maximum(candidate, floor)

    def _guard_bid_level(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        negative_inf: torch.Tensor,
    ) -> torch.Tensor:
        ceiling = torch.nextafter(
            (previous - self.gap_eps).detach(),
            negative_inf,
        )
        return torch.minimum(candidate, ceiling)

    def decode(self, params: torch.Tensor) -> torch.Tensor:
        """Decode unconstrained params to a valid [B, 2, 20, T] LOB tensor."""
        if params.ndim != 4 or params.shape[1] != 2 or params.shape[2] != 20:
            raise ValueError(f"Expected [B, 2, 20, T], got {tuple(params.shape)}")

        price_params = params[:, 0]
        volume_params = params[:, 1]

        mid = price_params[:, 0:1, :]
        spread = F.softplus(price_params[:, 1:2, :]) + self.min_spread
        ask_gaps = self._decode_gap_excess(price_params[:, 2:11, :]) + self.gap_eps
        bid_gaps = self._decode_gap_excess(price_params[:, 11:20, :]) + self.gap_eps

        bid1 = mid - 0.5 * spread
        positive_inf = torch.full_like(bid1, float("inf"))
        negative_inf = torch.full_like(bid1, float("-inf"))
        ask1_candidate = mid + 0.5 * spread
        ask1_floor = torch.nextafter(
            (bid1 + self.min_spread).detach(),
            positive_inf,
        )
        ask1 = torch.maximum(ask1_candidate, ask1_floor)

        # Cumulative sums can lose a small minimum gap when a tiny increment is
        # added to a larger float32 midpoint. Build each level sequentially and
        # use the next representable value above/below the hard boundary.
        ask_levels = [ask1]
        for idx in range(ask_gaps.shape[1]):
            previous = ask_levels[-1]
            candidate = previous + ask_gaps[:, idx:idx + 1]
            ask_levels.append(self._guard_ask_level(previous, candidate, positive_inf))

        bid_levels = [bid1]
        for idx in range(bid_gaps.shape[1]):
            previous = bid_levels[-1]
            candidate = previous - bid_gaps[:, idx:idx + 1]
            bid_levels.append(self._guard_bid_level(previous, candidate, negative_inf))

        prices = torch.cat(ask_levels + bid_levels, dim=1)
        volumes = F.softplus(volume_params) + self.volume_eps
        return torch.stack([prices, volumes], dim=1)

    def encode(self, lob: torch.Tensor) -> torch.Tensor:
        """Encode a valid LOB tensor into unconstrained transform coordinates."""
        if lob.ndim != 4 or lob.shape[1] != 2 or lob.shape[2] != 20:
            raise ValueError(f"Expected [B, 2, 20, T], got {tuple(lob.shape)}")

        prices = lob[:, 0]
        volumes = lob[:, 1]
        ask = prices[:, :10]
        bid = prices[:, 10:]

        mid = (ask[:, 0:1, :] + bid[:, 0:1, :]) / 2
        spread = ask[:, 0:1, :] - bid[:, 0:1, :]
        ask_gaps = ask[:, 1:, :] - ask[:, :-1, :]
        bid_gaps = bid[:, :-1, :] - bid[:, 1:, :]

        spread_raw = _inverse_softplus(torch.clamp(spread - self.min_spread, min=1e-12))
        ask_gap_raw = self._encode_gap_excess(ask_gaps - self.gap_eps)
        bid_gap_raw = self._encode_gap_excess(bid_gaps - self.gap_eps)
        volume_raw = _inverse_softplus(torch.clamp(volumes - self.volume_eps, min=1e-12))

        price_params = torch.cat([mid, spread_raw, ask_gap_raw, bid_gap_raw], dim=1)
        return torch.stack([price_params, volume_raw], dim=1)


@dataclass(frozen=True)
class BoundaryAwareLegalLOBTransform(LegalLOBTransform):
    """LegalLOB variant that can represent minimum-gap boundary atoms.

    Spread and depth retain the smooth softplus links. Ask/bid gap excess uses
    ``relu(u)^2`` so finite coordinates with ``u <= 0`` decode to the semantic
    lower-bound gap in real arithmetic. The map reaches the closed gap
    constraint set but is not one-to-one. Floating-point reconstruction may add
    a local representational guard if absolute-price addition rounds below the
    requested minimum increment.
    """

    def _decode_gap_excess(self, params: torch.Tensor) -> torch.Tensor:
        return torch.relu(params).square()

    def _encode_gap_excess(self, excess: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(excess, min=0.0))

    def _guard_ask_level(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        positive_inf: torch.Tensor,
    ) -> torch.Tensor:
        boundary = (previous + self.gap_eps).detach()
        guarded = torch.maximum(candidate, boundary)
        rounded_short = guarded - previous < self.gap_eps
        next_boundary = torch.nextafter(boundary, positive_inf)
        return torch.where(rounded_short, next_boundary, guarded)

    def _guard_bid_level(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        negative_inf: torch.Tensor,
    ) -> torch.Tensor:
        boundary = (previous - self.gap_eps).detach()
        guarded = torch.minimum(candidate, boundary)
        rounded_short = previous - guarded < self.gap_eps
        next_boundary = torch.nextafter(boundary, negative_inf)
        return torch.where(rounded_short, next_boundary, guarded)


@dataclass(frozen=True)
class MatchedLOBTransform:
    """Unconstrained baseline in the same semantic LOB coordinates.

    The coordinate rows match :class:`LegalLOBTransform` conceptually:
    midpoint, spread, ask gaps, bid gaps, and volumes. Unlike LegalLOB, this
    transform uses identity links for spread, gaps, and volumes. Generated
    values may therefore be negative and decoded books are not guaranteed to
    be valid. This isolates the benefit of coordinate choice from the benefit
    of positivity-preserving links.
    """

    def decode(self, params: torch.Tensor) -> torch.Tensor:
        if params.ndim != 4 or params.shape[1] != 2 or params.shape[2] != 20:
            raise ValueError(f"Expected [B, 2, 20, T], got {tuple(params.shape)}")

        price_params = params[:, 0]
        mid = price_params[:, 0:1]
        spread = price_params[:, 1:2]
        ask_gaps = price_params[:, 2:11]
        bid_gaps = price_params[:, 11:20]

        ask1 = mid + 0.5 * spread
        bid1 = mid - 0.5 * spread
        ask_tail = ask1 + torch.cumsum(ask_gaps, dim=1)
        bid_tail = bid1 - torch.cumsum(bid_gaps, dim=1)
        prices = torch.cat([ask1, ask_tail, bid1, bid_tail], dim=1)
        return torch.stack([prices, params[:, 1]], dim=1)

    def encode(self, lob: torch.Tensor) -> torch.Tensor:
        if lob.ndim != 4 or lob.shape[1] != 2 or lob.shape[2] != 20:
            raise ValueError(f"Expected [B, 2, 20, T], got {tuple(lob.shape)}")

        prices = lob[:, 0]
        ask = prices[:, :10]
        bid = prices[:, 10:]
        mid = (ask[:, 0:1] + bid[:, 0:1]) / 2
        spread = ask[:, 0:1] - bid[:, 0:1]
        ask_gaps = ask[:, 1:] - ask[:, :-1]
        bid_gaps = bid[:, :-1] - bid[:, 1:]
        price_params = torch.cat([mid, spread, ask_gaps, bid_gaps], dim=1)
        return torch.stack([price_params, lob[:, 1]], dim=1)


def make_lob_transform(
    representation: str,
    min_spread: float = 1e-6,
    gap_eps: float = 1e-8,
    volume_eps: float = 0.0,
) -> LegalLOBTransform | BoundaryAwareLegalLOBTransform | MatchedLOBTransform:
    if representation == "legal_lob":
        return LegalLOBTransform(min_spread, gap_eps, volume_eps)
    if representation == "legal_lob_ba":
        return BoundaryAwareLegalLOBTransform(min_spread, gap_eps, volume_eps)
    if representation == "matched_lob":
        return MatchedLOBTransform()
    raise ValueError(f"Representation {representation!r} has no coordinate transform")


def encode_lob(
    x: torch.Tensor,
    min_spread: float = 1e-6,
    gap_eps: float = 1e-8,
    volume_eps: float = 0.0,
) -> torch.Tensor:
    """Functional wrapper for LegalLOBTransform.encode."""
    return LegalLOBTransform(min_spread, gap_eps, volume_eps).encode(x)


def decode_lob(
    z: torch.Tensor,
    min_spread: float = 1e-6,
    gap_eps: float = 1e-8,
    volume_eps: float = 0.0,
) -> torch.Tensor:
    """Functional wrapper for LegalLOBTransform.decode."""
    return LegalLOBTransform(min_spread, gap_eps, volume_eps).decode(z)
