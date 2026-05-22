"""In-memory store + query helpers backing mantis-boattrader.

Loaded once at boot from seed.build(). All catalog access goes through
helpers here so route handlers stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import seed
from .seed import AdCreative, Boat, Dealer


# ---------------------------------------------------------------------------
# Boot-time state
# ---------------------------------------------------------------------------


@dataclass
class Store:
    boats: list[Boat]
    boats_by_id: dict[str, Boat]
    boats_by_slug: dict[str, Boat]
    dealers: list[Dealer]
    dealers_by_id: dict[str, Dealer]
    ads: list[AdCreative]
    facets: dict
    # Lead-form submissions persisted in-process for the session.
    leads: list[dict]
    # Audit log of every state-changing operation — drives the
    # ``/__env__/oracle`` graders so they can reconstruct what the
    # agent did even after the fact. Append-only; cleared on reset.
    mutations: list[dict]


_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        built = seed.build()
        boats: list[Boat] = built["boats"]
        dealers: list[Dealer] = built["dealers"]
        _store = Store(
            boats=boats,
            boats_by_id={b.id: b for b in boats},
            boats_by_slug={b.slug: b for b in boats},
            dealers=dealers,
            dealers_by_id={d.id: d for d in dealers},
            ads=built["ads"],
            facets=seed.facet_counts(boats),
            leads=[],
            mutations=[],
        )
    return _store


def reset() -> None:
    """Reseed — used by ``/__env__/reset``."""
    global _store
    _store = None
    store()


# ---------------------------------------------------------------------------
# Listing query
# ---------------------------------------------------------------------------


SORTS: list[tuple[str, str]] = [
    ("recommended", "Recommended"),
    ("price-asc", "Price: Low to High"),
    ("price-desc", "Price: High to Low"),
    ("year-desc", "Year: Newest First"),
    ("year-asc", "Year: Oldest First"),
    ("length-desc", "Length: Longest First"),
    ("length-asc", "Length: Shortest First"),
    ("newest", "Newly Listed"),
]


def _maybe_int(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _maybe_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def query_boats(params: dict, page: int = 1, per_page: int = 24) -> dict:
    """Apply filters from a flat querystring-style mapping. Returns the page
    slice plus the filtered total + the resolved filter values so the
    template can re-render the sidebar in its current state.
    """
    s = store()
    boats = s.boats

    type_v = params.get("type")
    make_v = params.get("make")
    cond_v = params.get("condition")
    state_v = params.get("state")
    city_v = params.get("city")
    q = (params.get("q") or "").strip().lower()

    year_min = _maybe_int(params.get("year_min"))
    year_max = _maybe_int(params.get("year_max"))
    length_min = _maybe_float(params.get("length_min"))
    length_max = _maybe_float(params.get("length_max"))
    price_min = _maybe_int(params.get("price_min"))
    price_max = _maybe_int(params.get("price_max"))
    price_drop = params.get("price_drop") in {"1", "true", "on"}

    def keep(b: Boat) -> bool:
        if type_v and b.boat_type != type_v:
            return False
        if make_v and b.make != make_v:
            return False
        if cond_v and cond_v != "all" and b.condition != cond_v:
            return False
        if state_v and b.state != state_v:
            return False
        if city_v and b.city.lower() != city_v.lower():
            return False
        if year_min is not None and b.year < year_min:
            return False
        if year_max is not None and b.year > year_max:
            return False
        if length_min is not None and b.length_ft < length_min:
            return False
        if length_max is not None and b.length_ft > length_max:
            return False
        if price_min is not None and (b.price is None or b.price < price_min):
            return False
        if price_max is not None and (b.price is None or b.price > price_max):
            return False
        if price_drop and "Price Drop" not in b.badges:
            return False
        if q:
            hay = f"{b.title} {b.boat_type} {b.city} {b.state}".lower()
            if q not in hay:
                return False
        return True

    filtered = [b for b in boats if keep(b)]

    sort_v = params.get("sort") or "recommended"
    if sort_v == "price-asc":
        filtered.sort(key=lambda b: (b.price is None, b.price or 0))
    elif sort_v == "price-desc":
        filtered.sort(key=lambda b: (b.price is None, -(b.price or 0)))
    elif sort_v == "year-desc":
        filtered.sort(key=lambda b: -b.year)
    elif sort_v == "year-asc":
        filtered.sort(key=lambda b: b.year)
    elif sort_v == "length-desc":
        filtered.sort(key=lambda b: -b.length_ft)
    elif sort_v == "length-asc":
        filtered.sort(key=lambda b: b.length_ft)
    elif sort_v == "newest":
        filtered.sort(key=lambda b: b.listed_days_ago)
    else:  # recommended — pin Featured first, then by views desc
        filtered.sort(key=lambda b: (0 if "Featured" in b.badges else 1, -b.views))

    total = len(filtered)
    pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, pages))
    start = (page - 1) * per_page
    page_items = filtered[start:start + per_page]

    return {
        "boats": page_items,
        "total": total,
        "page": page,
        "pages": pages,
        "per_page": per_page,
        "sort": sort_v,
        "facets": s.facets,
        "filters": {
            "type": type_v,
            "make": make_v,
            "condition": cond_v or "all",
            "state": state_v,
            "city": city_v,
            "q": q,
            "year_min": year_min,
            "year_max": year_max,
            "length_min": length_min,
            "length_max": length_max,
            "price_min": price_min,
            "price_max": price_max,
            "price_drop": price_drop,
        },
    }


def featured_boats(limit: int = 12) -> list[Boat]:
    s = store()
    feat = [b for b in s.boats if "Featured" in b.badges]
    feat.sort(key=lambda b: -b.views)
    return feat[:limit]


def boats_near(_zip: str | None, limit: int = 8) -> list[Boat]:
    s = store()
    # The "Near You" rail mimics a coastal rotation — same set every load.
    rotation = sorted(s.boats, key=lambda b: (b.listed_days_ago, -b.views))[:limit]
    return rotation


def popular_types() -> list[tuple[str, int, str]]:
    s = store()
    type_to_image = {
        "Runabout": "/assets/img/type_runabout.svg",
        "Center Console": "/assets/img/type_centerconsole.svg",
        "Pontoon": "/assets/img/type_pontoon.svg",
        "Bowrider": "/assets/img/type_bowrider.svg",
        "Cruiser": "/assets/img/type_cruiser.svg",
        "Sailboat": "/assets/img/type_sailboat.svg",
        "Yacht": "/assets/img/type_yacht.svg",
        "Trawler": "/assets/img/type_trawler.svg",
        "Bass": "/assets/img/type_bass.svg",
        "Personal Watercraft": "/assets/img/type_pwc.svg",
    }
    out: list[tuple[str, int, str]] = []
    for t, n in s.facets["by_type"].items():
        out.append((t, n, type_to_image.get(t, "/assets/img/type_runabout.svg")))
    return out


def list_makes() -> list[tuple[str, int]]:
    return list(store().facets["by_make"].items())


def boat_by_slug(slug: str) -> Boat | None:
    return store().boats_by_slug.get(slug)


def dealer_for(boat: Boat) -> Dealer:
    return store().dealers_by_id[boat.dealer_id]


def adjacent_boats(boat: Boat) -> tuple[Boat | None, Boat | None]:
    s = store()
    if boat.id not in s.boats_by_id:
        return None, None
    boats = s.boats
    idx = next(i for i, b in enumerate(boats) if b.id == boat.id)
    prev = boats[idx - 1] if idx > 0 else None
    nxt = boats[idx + 1] if idx + 1 < len(boats) else None
    return prev, nxt


def similar_boats(boat: Boat, limit: int = 6) -> list[Boat]:
    s = store()
    same_type = [b for b in s.boats if b.boat_type == boat.boat_type and b.id != boat.id]
    same_type.sort(key=lambda b: (abs(b.length_ft - boat.length_ft), abs(b.year - boat.year)))
    return same_type[:limit]


def record_lead(payload: dict) -> dict:
    s = store()
    entry = {"id": f"lead-{len(s.leads) + 1:05d}", **payload}
    s.leads.append(entry)
    emit_mutation(
        operation="lead_submitted",
        target_type="boat",
        target_id=str(payload.get("boat_id") or ""),
        payload={k: v for k, v in entry.items() if k != "ts"},
    )
    return entry


def list_leads() -> list[dict]:
    return list(store().leads)


# ---------------------------------------------------------------------------
# Mutations audit log
# ---------------------------------------------------------------------------


def emit_mutation(
    *,
    operation: str,
    target_type: str,
    target_id: str,
    payload: dict | None = None,
) -> dict:
    """Append a single mutation entry to the audit log.

    Oracles read this log to grade what the agent actually did — same
    shape as the SQLite ``mutations`` table used by the CRM / shop /
    helpdesk envs, but kept in memory here since boattrader's store is
    in-memory too.
    """
    s = store()
    entry = {
        "id": len(s.mutations) + 1,
        "operation": operation,
        "target_type": target_type,
        "target_id": target_id,
        "payload": dict(payload or {}),
    }
    s.mutations.append(entry)
    return entry


def list_mutations(*, limit: int | None = None) -> list[dict]:
    """Return mutations in chronological order. ``limit`` returns the tail."""
    muts = store().mutations
    if limit is None or limit >= len(muts):
        return list(muts)
    return list(muts[-limit:])
