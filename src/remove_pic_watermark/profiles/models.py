"""Watermark profile domain models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ProfileKind(str, Enum):
    """Internal detector kind (kept for compatibility).

    Product UI treats all user-created styles as one「水印样式」;
    batch strategy is controlled by MatchStrategy.
    """

    TEMPLATE = "template"
    FIXED_BOX = "fixed_box"


class MatchStrategy(str, Enum):
    """How to locate this style on new images during batch."""

    AUTO = "auto"  # legacy → treated as FOLLOW in UI
    PIN = "pin"  # hard-paste mask at recorded position (no matching)
    FOLLOW = "follow"  # match near sample ROI
    SEARCH = "search"  # full-frame match


@dataclass
class Profile:
    """One reusable watermark style."""

    id: str
    name: str
    kind: ProfileKind
    enabled: bool = True
    description: str = ""
    template_file: str | None = None
    # detector parameters (template or fixed_box)
    detector: dict[str, Any] = field(default_factory=dict)
    # provenance for UI / debugging
    created_from: dict[str, Any] = field(default_factory=dict)
    # batch locate strategy (product-facing)
    match_strategy: MatchStrategy = MatchStrategy.AUTO
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["match_strategy"] = self.match_strategy.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Profile":
        kind_value = data.get("kind", ProfileKind.TEMPLATE.value)
        strategy_raw = data.get("match_strategy") or data.get("strategy") or MatchStrategy.AUTO.value
        # legacy fixed_box defaults to follow
        if strategy_raw == MatchStrategy.AUTO.value and kind_value == ProfileKind.FIXED_BOX.value:
            strategy_raw = MatchStrategy.FOLLOW.value
        try:
            strategy = MatchStrategy(strategy_raw)
        except ValueError:
            strategy = MatchStrategy.AUTO
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or data["id"]),
            kind=ProfileKind(kind_value),
            enabled=bool(data.get("enabled", True)),
            description=str(data.get("description") or ""),
            template_file=data.get("template_file"),
            detector=dict(data.get("detector") or {}),
            created_from=dict(data.get("created_from") or {}),
            match_strategy=strategy,
            version=int(data.get("version", 1)),
        )

    def template_path(self, profile_dir: Path) -> Path | None:
        if not self.template_file:
            return None
        return profile_dir / self.template_file

    def strategy_label(self) -> str:
        return {
            MatchStrategy.AUTO: "附近匹配",
            MatchStrategy.PIN: "固定位置",
            MatchStrategy.FOLLOW: "附近匹配",
            MatchStrategy.SEARCH: "全图匹配",
        }.get(self.match_strategy, "附近匹配")
