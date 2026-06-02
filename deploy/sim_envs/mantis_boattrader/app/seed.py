"""Deterministic seed for mantis-boattrader.

Generates a closed catalog of ~600 boats across ~9 boat types, ~25 makes,
hosted by ~30 dealers across US coastal/lake markets. Static — no network
required at boot.

Tunable via env vars:

* ``SEED`` (int, default ``42``) — RNG seed.
* ``BOAT_COUNT`` (int, default ``600``) — total listings.
* ``FAKE_NOW`` (iso, default ``2026-01-15T09:00:00Z``) — "today" for
  listing timestamps and "X days ago" computations.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

FAKE_NOW_DEFAULT = "2026-01-15T09:00:00Z"


BOAT_TYPES: list[str] = [
    "Runabout",
    "Center Console",
    "Pontoon",
    "Bowrider",
    "Cruiser",
    "Sailboat",
    "Yacht",
    "Trawler",
    "Bass",
    "Personal Watercraft",
]

# (make, models[]) — picked to match real Boat Trader top makes.
MAKES: dict[str, list[str]] = {
    "Boston Whaler": ["130 Super Sport", "150 Montauk", "170 Montauk", "210 Dauntless", "270 Vantage"],
    "Sea Ray": ["190 SPX", "230 SLX", "250 SDX", "320 Sundancer", "400 Sundancer"],
    "Bennington": ["20 SVSR", "22 MSB", "23 LSR", "25 RSB"],
    "Bayliner": ["VR4", "VR5", "DX2050", "DX2250"],
    "Grady-White": ["Freedom 215", "Canyon 271", "Canyon 366", "Express 370"],
    "Regulator": ["28", "31", "34", "41"],
    "Yamaha": ["WaveRunner FX HO", "WaveRunner VX Cruiser", "242 Limited S"],
    "Cobalt": ["R3", "R5", "R7", "R8", "R30"],
    "Chaparral": ["19 SSi", "23 SSi", "267 SSX", "327 SSX"],
    "Robalo": ["R200", "R230", "R272"],
    "Tracker": ["Pro Team 175 TXW", "Pro Guide V-16 SC", "Targa V-19 Combo"],
    "MasterCraft": ["NXT22", "X22", "XStar"],
    "Nautique": ["Super Air G23 Paragon", "Super Air G25"],
    "Catalina": ["355", "385", "425"],
    "Beneteau": ["Oceanis 34.1", "Oceanis 40.1", "Antares 7"],
    "Jeanneau": ["Sun Odyssey 410", "Cap Camarat 9.0 WA"],
    "Hatteras": ["GT54", "GT63", "M75"],
    "Viking": ["52 Open", "68 Convertible", "92 Sky"],
    "Princess": ["F55", "Y72", "Y85"],
    "Azimut": ["50 Fly", "Magellano 60", "S7"],
    "Pursuit": ["S 268", "S 328", "OS 385"],
    "Crestliner": ["VT 18", "1850 Bass Hawk", "2050 Sportfish"],
    "Lund": ["1875 Pro-V", "208 Pro-V GL"],
    "Albin": ["28 Tournament Express", "30 Family Cruiser"],
    "Alm": ["Delfino 64 Trawler", "Delfino 76 Trawler"],
}

TYPE_BY_MAKE: dict[str, list[str]] = {
    "Boston Whaler": ["Center Console", "Runabout"],
    "Sea Ray": ["Bowrider", "Cruiser", "Yacht"],
    "Bennington": ["Pontoon"],
    "Bayliner": ["Bowrider", "Runabout"],
    "Grady-White": ["Center Console", "Cruiser"],
    "Regulator": ["Center Console"],
    "Yamaha": ["Personal Watercraft", "Bowrider"],
    "Cobalt": ["Bowrider", "Cruiser"],
    "Chaparral": ["Bowrider", "Cruiser"],
    "Robalo": ["Center Console"],
    "Tracker": ["Bass"],
    "MasterCraft": ["Runabout"],
    "Nautique": ["Runabout"],
    "Catalina": ["Sailboat"],
    "Beneteau": ["Sailboat"],
    "Jeanneau": ["Sailboat"],
    "Hatteras": ["Yacht"],
    "Viking": ["Yacht"],
    "Princess": ["Yacht"],
    "Azimut": ["Yacht"],
    "Pursuit": ["Cruiser", "Center Console"],
    "Crestliner": ["Bass", "Runabout"],
    "Lund": ["Bass"],
    "Albin": ["Trawler"],
    "Alm": ["Trawler"],
}


LOCATIONS: list[tuple[str, str, str]] = [
    ("Fort Lauderdale", "FL", "33316"),
    ("Miami", "FL", "33132"),
    ("Naples", "FL", "34102"),
    ("Tampa", "FL", "33606"),
    ("Sarasota", "FL", "34236"),
    ("Jacksonville", "FL", "32202"),
    ("Charleston", "SC", "29401"),
    ("Annapolis", "MD", "21401"),
    ("Norfolk", "VA", "23510"),
    ("Newport", "RI", "02840"),
    ("Boston", "MA", "02110"),
    ("Portland", "ME", "04101"),
    ("New York", "NY", "10004"),
    ("Long Beach", "NY", "11561"),
    ("San Diego", "CA", "92101"),
    ("Newport Beach", "CA", "92660"),
    ("Marina del Rey", "CA", "90292"),
    ("Sausalito", "CA", "94965"),
    ("Seattle", "WA", "98101"),
    ("Anacortes", "WA", "98221"),
    ("Portland", "OR", "97201"),
    ("Galveston", "TX", "77550"),
    ("Houston", "TX", "77002"),
    ("Lake Charles", "LA", "70601"),
    ("Chicago", "IL", "60601"),
    ("Lake Geneva", "WI", "53147"),
    ("Lake Tahoe", "CA", "96150"),
    ("Lake Norman", "NC", "28117"),
    ("Lake of the Ozarks", "MO", "65049"),
    ("Campbell", "CA", "95008"),
]

DEALER_PREFIXES: list[str] = [
    "Sun Country Marine Group",
    "Allied Marine",
    "Nauti Yacht Sales",
    "Galati Yacht Sales",
    "MarineMax",
    "Boats Inc.",
    "Coastal Cruising",
    "Harbor East Marine",
    "Pacific Boating Group",
    "Atlantic Yacht Brokerage",
    "Tidewater Marine",
    "Anchor Bay Yachts",
    "Marina Bay Boats",
    "Boatworld",
    "Premier Boating",
    "United Yacht Sales",
    "Bluewater Yacht Sales",
    "OneWater Marine",
    "Off the Hook Yachts",
    "South Coast Marine",
]

HULL_MATERIAL = ["Fiberglass", "Aluminum", "Fiberglass / GRP", "Composite"]
FUEL_TYPE = ["Gasoline", "Diesel", "Electric"]
# Engine make is assigned by hull plausibility, not uniformly: Caterpillar
# and Cummins are big marine diesels found on yachts / trawlers / large
# cruisers, never on a 16ft bass boat or a PWC. The two pools below keep the
# Caterpillar target (the BT02 oracle) on realistic hulls while preserving the
# "must open the detail page" property — within a diesel hull the make is a
# 3-way mix (Caterpillar / Cummins / Volvo Penta), so boat type narrows the
# search but does NOT determine the engine. Volvo Penta straddles both pools
# (gas sterndrives + diesel inboards), which also yields same-make decoys.
DIESEL_ENGINE_MAKES = ["Caterpillar", "Cummins", "Volvo Penta"]
GAS_ENGINE_MAKES = ["Mercury", "Yamaha", "Suzuki", "Honda", "Volvo Penta"]
DRIVE_TYPE = ["Outboard", "Sterndrive", "Inboard", "Jet drive"]


def _is_diesel_hull(boat_type: str, length_ft: float) -> bool:
    """Hulls that realistically carry a big marine diesel (Caterpillar/Cummins).

    Yachts and trawlers always do; cruisers and center consoles only once
    they're large enough (>= 40ft, the same threshold the fuel-type logic
    uses to mark a boat Diesel). Sailboats are excluded — they get a small
    Yanmar auxiliary, not a Caterpillar.
    """
    if boat_type in ("Yacht", "Trawler"):
        return True
    if boat_type in ("Cruiser", "Center Console") and length_ft >= 40:
        return True
    return False

# Each entry is (opener, build_section, features_section, history_section).
# Renderer joins with double newlines so it produces a multi-paragraph block.
DESCRIPTION_BLURBS: list[tuple[str, str, str, str]] = [
    (
        "Lightly used and meticulously maintained — a rare opportunity to step into a {year} {make} {model} at this condition level.",
        "Built on {make}'s signature deep-V hull with {hull_material} construction, this boat has been freshwater-only since new and stored indoors every off-season.",
        "Factory equipment includes premium JL Audio sound system, hardtop with rod holders, integrated bow thruster, electric windlass with chain counter, and a fully enclosed head with shower. The cockpit has been upgraded with SeaDek throughout and the upholstery is in show condition.",
        "Full service records available from the original delivering dealer. Annual service performed by an authorized {engine_make} technician. Survey-ready and turn-key.",
    ),
    (
        "Original-owner {year} {make} {model} with documented low engine hours and warranty still in effect.",
        "{hull_color} hull with the upgraded {engine_count}× {engine_make} power package. Joystick docking, dynamic positioning, and a Garmin glass-bridge electronics suite.",
        "Highlights include a full enclosure, generator, reverse-cycle climate control, watermaker, davits for the tender, and underwater LED lighting. Tower is rigged for tuna with outriggers and a livewell aft.",
        "Recently hauled for fresh bottom paint, new zincs, and a full systems inspection. Stored on a covered lift when not in use. Annual maintenance is current.",
    ),
    (
        "Custom-ordered {year} {make} {model} in the {hull_color} flag color with the optional performance package.",
        "Reinforced {hull_material} hull, transom-mounted hydraulic swim platform, and a fold-down arch for low-clearance bridges. Twin {engine_count}× {engine_make} engines provide an effortless cruise at 28 knots and a top end above 40.",
        "Loaded with the entertainment package: forward sun pad, transom grill, refrigerated drawer, and underwater music. Below deck includes a private berth, marine head, and a small galley with two-burner stove.",
        "Surveys and service records will be provided to the qualified buyer. Bottom is freshly painted; trailer is included and recently re-bearing'd.",
    ),
    (
        "Turn-key {year} {make} {model} ready for the season — slip in {city}, {state} can transfer with the sale.",
        "Hull: {hull_material} in {hull_color}. Power: {engine_count}× {engine_make} {engine_hp} hp with the long-shaft package. Drive: {drive_type}. Fuel: {fuel_type}.",
        "Equipment list: bow & stern thrusters, Raymarine Axiom multifunction displays, autopilot, AIS, radar, two refrigerators, dishwasher, washer/dryer, and a 17.5 kW Onan generator. Three staterooms, two heads with separate showers.",
        "Coast Guard compliant safety package included. The seller is moving to a larger boat and is highly motivated. Sea trials welcome; survey at buyer's expense.",
    ),
    (
        "Coastal cruising-ready {year} {make} {model} that has been the pride of one family since new.",
        "{hull_color} {hull_material} hull with the cruising package — bow & stern pulpit with stanchions, deck-mounted windlass, and dual-anchor configuration. Powered by {engine_count}× {engine_make} {engine_hp} hp ({drive_type}).",
        "Comfortable interior with full galley, dinette converts to berth, mid-cabin V-berth with privacy door, enclosed head with shower, and reverse-cycle A/C. Cockpit canvas, bimini, and full enclosure all in excellent condition.",
        "Coast Guard registered and insured. Recent survey on file — copy available to qualified buyers. Trailer is NOT included; can be arranged separately.",
    ),
]


OWNER_DESCRIPTION_BLURBS: list[tuple[str, str, str, str]] = [
    (
        "Selling my {year} {make} {model} — meticulously maintained and ready for her next family.",
        "I purchased her new and she's been freshwater-only her whole life. {hull_color} hull, {engine_count}× {engine_make} power. {hours} hours on the engines.",
        "Comes with everything you need to splash and enjoy: full canvas, custom mooring cover, fenders, lines, life jackets, and a stocked safety kit. Recent service paperwork is in a binder in the cabin.",
        "I'm reluctantly selling because I'm moving inland for work. Cash deal preferred. Will negotiate the trailer separately if needed. Serious inquiries only — boat is in {city}, {state}.",
    ),
    (
        "Original owner — bought new from the dealer, never chartered, never raced. {year} {make} {model}.",
        "Single-owner boat. {hull_material} hull in {hull_color}. {engine_count}× {engine_make} {engine_hp} hp with documented engine hours of {hours}.",
        "Recently completed service includes new impellers, fresh oil and filters, new anodes, and a full electronics calibration. Bottom paint is one season old.",
        "Boat is in our slip in {city}, {state}. Available for viewing weekends — please reach out via the form. Survey welcome at buyer's expense.",
    ),
]

OWNER_HIGHLIGHTS: list[list[str]] = [
    ["Easy to handle", "Great fuel economy", "Spacious cockpit"],
    ["Smooth ride", "Quiet at cruise", "Excellent visibility"],
    ["Plenty of storage", "Comfortable cabin", "Holds value well"],
    ["Stable at anchor", "Family-friendly layout", "Reliable engines"],
]


# Short pill-tags shown in the "Owner Highlights" column of the
# "What Owners Say" card. Drawn from a closed vocabulary so the
# What-Owners-Say paragraph below can name them by reference.
OWNER_HIGHLIGHT_TAGS_POOL: list[str] = [
    "reliable performance", "strong power", "responsive performance",
    "confident handling", "longer trip capability", "spacious layout",
    "fuel efficient", "great resale value", "quiet at cruise",
    "smooth ride", "excellent visibility", "easy single-handed docking",
    "family friendly", "low maintenance", "well-mannered in chop",
]


# "What Owners Say" templates — mimic the AI-summary card on real
# boattrader.com (the sparkle/AI icon + paragraph + info tooltip).
WHAT_OWNERS_SAY_TEMPLATES: list[str] = [
    (
        "The {make} {model} is renowned for its {tag1} and {tag2}, making it a "
        "favorite among boating enthusiasts. Its {engine_count_word} {engine_make} "
        "engine{engine_plural} deliver{engine_verb_s} {tag3}, ensuring an exhilarating "
        "experience on the water. The boat's design offers {tag4}, allowing for "
        "smooth navigation in various conditions. With a seating capacity of "
        "{capacity}, it comfortably accommodates groups, enhancing its appeal for "
        "social outings. Additionally, its {fuel_capacity} fuel capacity supports "
        "{tag5}, adding practical convenience for extended adventures."
    ),
    (
        "Owners consistently praise the {make} {model} for its {tag1} and {tag2}, "
        "noting that the {hull_material} construction holds up exceptionally well "
        "over time. Powered by {engine_count_word} {engine_make} engine{engine_plural}, "
        "this {length_ft}-foot {boat_type} delivers {tag3} while remaining "
        "approachable for newer captains. The {tag4} layout — combined with seating "
        "for {capacity} — makes it equally at home for weekend cruises and "
        "all-day water sports. A {fuel_capacity}-gallon fuel tank backs up the "
        "{tag5} owners frequently mention in reviews."
    ),
    (
        "Boaters who own the {make} {model} highlight {tag1}, {tag2}, and {tag3} "
        "as standout characteristics. The {drive_type} configuration paired with "
        "the {engine_make} powerplant has earned a reputation for {tag4} across "
        "the ownership community. At {length_ft} feet with passenger capacity of "
        "{capacity}, it strikes a balance between performance and usability. Its "
        "{fuel_capacity}-gallon tank lends itself to {tag5}, which owners cite "
        "when comparing it to similarly-priced rivals."
    ),
]


# Original-seller "Description" blurbs that include the legacy "?s"
# encoding glitches commonly seen on real boattrader listings (the
# straight apostrophe being mangled into "?" during a CMS migration).
DEALER_SELLER_DESCRIPTIONS: list[str] = [
    (
        "The {model} features a long, firm, rampy wake, with a clean lip and no "
        "trough. The stern seat affords three different configurations so you can "
        "customize your boat's interior to suit you and {seat_n} of your closest "
        "friends. In its original position it completes an expansive stern seat. "
        "Remove the center section and position it midship and you have an aft "
        "facing observer's seat. Turn it around and you have a forward facing "
        "companion seat. It's up to you."
    ),
    (
        "This {year} {make} {model} represents the pinnacle of {make}'s offshore "
        "lineup. Hand-laid {hull_material} construction with a deep-V hull pattern "
        "deliver a dry, comfortable ride at any speed. The forward seating module "
        "converts into a sun-pad and there's an integrated cooler underneath. The "
        "helm is sport-fishing focused but doesn't sacrifice creature comforts — "
        "you'll find a removable galley insert, refrigerator drawer, and freshwater "
        "spigot all within arm's reach of the captain."
    ),
    (
        "We're pleased to offer this clean and well-equipped {year} {make} {model}. "
        "It's a one-owner boat that's been impeccably maintained — all service "
        "records will transfer with the boat. The {hull_color} hull is in show "
        "condition; gelcoat is in remarkable shape for the year. Below deck you'll "
        "find {capacity} accommodations including a private master, and an enclosed "
        "head with separate stall shower. Won't last long — schedule a sea trial today."
    ),
    (
        "Don't miss this opportunity! This {make} {model} won't last on the brokerage "
        "market. She's been owner-operated since new and is being offered turn-key "
        "with everything you need to step aboard and enjoy. The galley is fully "
        "outfitted, the electronics suite has been refreshed within the last year, "
        "and the {engine_make} engine{engine_plural} {engine_verb_pres} just been "
        "serviced. Whether you're a first-time buyer or a seasoned captain, you'll "
        "appreciate the attention to detail."
    ),
]

HULL_COLORS = ["Diamond White", "Midnight Blue", "Slate Grey", "Ice Blue", "Pearl", "Carbon Black"]


@dataclass
class Dealer:
    id: str
    name: str
    address: str
    city: str
    state: str
    zip: str
    phone: str
    years_partner: int
    logo_initial: str
    active_listings: int
    sold_listings: int

    @property
    def trusted_partner_label(self) -> str:
        if self.years_partner >= 10:
            return "10+ years"
        return f"{self.years_partner} years"


@dataclass
class Boat:
    id: str
    slug: str
    year: int
    make: str
    model: str
    boat_type: str
    condition: str  # "new" | "used"
    length_ft: float
    beam_ft: float
    price: int | None  # None → "Request a Price"
    monthly_price: int | None
    original_price: int | None  # Set when this is a Price Drop listing.
    city: str
    state: str
    zip: str
    dealer_id: str  # populated for dealer + sponsored listings.
    listing_type: str  # "sponsored" | "dealer" | "owner"
    owner_name: str | None  # set for listing_type == "owner".
    owner_phone: str | None  # raw digits, only rendered via SVG (no plain text in HTML).
    hull_color: str
    hull_material: str
    fuel_type: str
    drive_type: str
    engine_make: str
    engine_hp: int
    engine_count: int
    hours: int  # used → real hours, new → 0
    capacity_people: int  # passenger capacity from CG cert / OEM spec.
    fuel_capacity_gal: int  # tank size — drives the "longer trip capability" tag.
    description: list[str]  # paragraphs (Description section).
    description_features: list[str]  # bullet list of key features.
    what_owners_say: str  # AI-style summary shown in the "What Owners Say" card.
    owner_highlight_tags: list[str]  # short pill-style tags shown next to the card.
    owner_seller_description: str  # original seller's blurb with classic CMS "?" entity quirks.
    owner_highlights: list[str]
    badges: list[str]
    views: int
    saves: int
    listed_days_ago: int
    images: list[str] = field(default_factory=list)
    # BT03 layout-drift flag — set by ``_apply_byowner_reveal_drift`` (gated by
    # ``BT03_REVEAL_DRIFT``). When True the detail template de-emphasises and
    # relabels the phone-reveal control (drops the "phone" keyword), so a frozen
    # agent favours the prominent lead form; an S1 worked-reveal exemplar
    # re-finds the control by its action. Off by default → zero render change.
    reveal_drift: bool = False

    @property
    def title(self) -> str:
        return f"{self.year} {self.make} {self.model}"

    @property
    def display_price(self) -> str:
        return f"${self.price:,}" if self.price else "Request a Price"

    @property
    def display_original_price(self) -> str | None:
        if self.original_price is None:
            return None
        return f"${self.original_price:,}"

    @property
    def price_drop_amount(self) -> str | None:
        if self.original_price is None or self.price is None:
            return None
        delta = self.original_price - self.price
        if delta <= 0:
            return None
        return f"-${delta:,}"

    @property
    def display_monthly(self) -> str | None:
        if self.monthly_price is None:
            return None
        return f"${self.monthly_price:,}/mo*"

    @property
    def is_owner_listed(self) -> bool:
        return self.listing_type == "owner"

    @property
    def is_sponsored(self) -> bool:
        return self.listing_type == "sponsored"


@dataclass
class AdCreative:
    id: str
    sponsor: str
    headline: str
    subline: str
    cta: str
    bg_a: str  # left/top color
    bg_b: str  # right/bottom color
    accent: str  # text/badge accent
    href: str


# ---------------------------------------------------------------------------


def _rand(seed: int) -> random.Random:
    return random.Random(seed)


def _gen_dealers(rng: random.Random) -> list[Dealer]:
    dealers: list[Dealer] = []
    for idx, prefix in enumerate(DEALER_PREFIXES):
        loc = LOCATIONS[idx % len(LOCATIONS)]
        suffix_pool = ["", f", {loc[0]}", f" — {loc[1]}", f" of {loc[0]}"]
        suffix = suffix_pool[idx % len(suffix_pool)]
        name = f"{prefix}{suffix}".strip(", ").strip()
        phone_seed = 4000000 + idx * 18371
        # Real BT renders dealer phones as raw `+1XXXXXXXXXX` (E.164,
        # no parens/dashes). Match that format.
        phone = f"+1{rng.randint(200, 989)}{rng.randint(200, 989):03d}{phone_seed % 10000:04d}"
        years = rng.choice([3, 6, 8, 12, 15, 18, 22])
        dealers.append(
            Dealer(
                id=f"dealer-{idx + 1:02d}",
                name=name,
                address=f"{rng.randint(100, 9999)} {rng.choice(['Harbor', 'Marina', 'Ocean', 'Lake', 'Bay', 'Yacht'])} {rng.choice(['Blvd', 'Way', 'Ave', 'Dr', 'St'])}",
                city=loc[0],
                state=loc[1],
                zip=loc[2],
                phone=phone,
                years_partner=years,
                logo_initial=prefix.split()[0][0].upper(),
                active_listings=rng.randint(40, 480),
                sold_listings=rng.randint(160, 2600),
            )
        )
    return dealers


def _gen_boats(rng: random.Random, dealers: list[Dealer], count: int, now: datetime) -> list[Boat]:
    boats: list[Boat] = []
    makes = list(MAKES.keys())
    for i in range(count):
        make = makes[i % len(makes)]
        model = rng.choice(MAKES[make])
        boat_type = rng.choice(TYPE_BY_MAKE[make])
        year = rng.choice([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
        condition = "new" if year >= 2025 and rng.random() < 0.65 else "used"
        # Length by type
        length_by_type = {
            "Runabout": (16, 26),
            "Center Console": (20, 42),
            "Pontoon": (18, 28),
            "Bowrider": (18, 30),
            "Cruiser": (28, 50),
            "Sailboat": (24, 58),
            "Yacht": (45, 100),
            "Trawler": (32, 80),
            "Bass": (16, 22),
            "Personal Watercraft": (10, 13),
        }
        lo, hi = length_by_type[boat_type]
        length_ft = round(rng.uniform(lo, hi), 1)
        beam_ft = round(length_ft * rng.uniform(0.28, 0.36), 1)
        # Price by length/condition/type
        base = {
            "Runabout": 45000,
            "Center Console": 90000,
            "Pontoon": 55000,
            "Bowrider": 60000,
            "Cruiser": 220000,
            "Sailboat": 180000,
            "Yacht": 1_900_000,
            "Trawler": 750_000,
            "Bass": 35_000,
            "Personal Watercraft": 14_000,
        }[boat_type]
        price = int(base * (length_ft / lo) * (1.0 if condition == "new" else 0.65) * rng.uniform(0.85, 1.25))

        # Listing type — three real BT shapes:
        #   ~8% sponsored (dealer-paid, boosted placement)
        #   ~67% dealer    (standard dealer listing, phone shown plain)
        #   ~25% owner     (private seller, phone gated behind reveal)
        # Computed BEFORE the POA branch below because that branch reads
        # ``listing_type`` (owner listings always carry a price).
        r = rng.random()
        if r < 0.08:
            listing_type = "sponsored"
        elif r < 0.75:
            listing_type = "dealer"
        else:
            listing_type = "owner"

        # Some boats are POA (~10%, only dealers — owners always list a price)
        if rng.random() < 0.08 and listing_type != "owner":
            price_final: int | None = None
            monthly: int | None = None
        else:
            price_final = price - (price % 100)
            monthly = int(price_final / 220)  # rough monthly @ ~5.5%, 20yr
        loc = rng.choice(LOCATIONS)
        dealer = rng.choice(dealers)
        if boat_type == "Sailboat":
            engine_make = "Yanmar"
        elif _is_diesel_hull(boat_type, length_ft):
            engine_make = rng.choice(DIESEL_ENGINE_MAKES)
        else:
            engine_make = rng.choice(GAS_ENGINE_MAKES)
        engine_count = 1 if length_ft < 28 else (2 if length_ft < 55 else 3)
        engine_hp = int(length_ft * rng.uniform(11, 18))
        hours = 0 if condition == "new" else rng.randint(20, 950)
        listed_days_ago = rng.randint(0, 120)

        # Badges
        badges: list[str] = []
        is_price_drop = rng.random() < 0.18 and listing_type != "owner"  # owner rarely advertises drop
        if listing_type == "sponsored":
            badges.append("Sponsored")
        elif "Featured" not in badges and i % 11 == 0:
            badges.append("Featured")
        if listed_days_ago <= 7:
            badges.append("New Arrival")
        if is_price_drop:
            badges.append("Price Drop")

        hull_color = rng.choice(HULL_COLORS)
        # Pick description pool by listing type — owner uses the
        # first-person blurbs, dealer/sponsored use the polished ones.
        if listing_type == "owner":
            opener, build, features, history = rng.choice(OWNER_DESCRIPTION_BLURBS)
        else:
            opener, build, features, history = rng.choice(DESCRIPTION_BLURBS)
        ctx = dict(
            year=year, make=make, model=model, color=hull_color.lower(),
            hull_color=hull_color, hull_material=rng.choice(HULL_MATERIAL),
            engine_make=engine_make, engine_count=engine_count, engine_hp=engine_hp,
            drive_type=rng.choice(DRIVE_TYPE),
            fuel_type="Diesel" if (_is_diesel_hull(boat_type, length_ft) or length_ft >= 40) else "Gasoline",
            hours=hours, city=loc[0], state=loc[1],
        )
        description_paragraphs = [
            opener.format(**ctx),
            build.format(**ctx),
            features.format(**ctx),
            history.format(**ctx),
        ]
        # Key features bullet list (rendered as ul). Pulled from a
        # bounded vocabulary so deterministic per seed.
        feature_pool = [
            "Bow & stern thrusters", "Joystick docking", "Garmin Axiom MFD electronics",
            "Raymarine autopilot", "Underwater LED lighting", "JL Audio sound system",
            "Refrigerated cockpit drawer", "Hydraulic swim platform", "Generator (Onan 17.5 kW)",
            "Reverse-cycle A/C", "Bow & stern pulpit with stanchions", "Electric windlass",
            "Marine head with shower", "SeaDek cockpit flooring", "Full canvas + bimini",
            "Outriggers + tuna tower", "Bottom paint < 1 year", "Trailer included",
        ]
        # Drift the feature set deterministically by boat index.
        feat_n = rng.randint(5, 9)
        rng.shuffle(feature_pool)
        description_features = feature_pool[:feat_n]

        # Owner identity for owner listings.
        if listing_type == "owner":
            first_names = ["Jim", "Maria", "Carlos", "Aiko", "Robert", "Pat", "Diane",
                           "Marcus", "Liu", "Hannah", "Thomas", "Beatrice", "Ravi"]
            owner_name = rng.choice(first_names)
            owner_phone = f"{rng.randint(200, 989)}{rng.randint(200, 989):03d}{rng.randint(0, 9999):04d}"
        else:
            owner_name = None
            owner_phone = None

        # Capacity by type/length.
        cap_by_type_base = {
            "Runabout": 6, "Center Console": 8, "Pontoon": 12, "Bowrider": 8,
            "Cruiser": 10, "Sailboat": 6, "Yacht": 14, "Trawler": 8,
            "Bass": 4, "Personal Watercraft": 2,
        }
        capacity_people = max(2, int(cap_by_type_base[boat_type] + (length_ft - 20) * 0.4 + rng.uniform(-1, 2)))
        fuel_capacity_gal = max(8, int(length_ft * rng.uniform(2.0, 4.5)))

        # What Owners Say — pick 5 tags, then fill template.
        tags_pool = list(OWNER_HIGHLIGHT_TAGS_POOL)
        rng.shuffle(tags_pool)
        owner_highlight_tags = tags_pool[:5]
        wos_template = rng.choice(WHAT_OWNERS_SAY_TEMPLATES)
        engine_count_word = {1: "one", 2: "twin", 3: "triple"}.get(engine_count, f"{engine_count}")
        what_owners_say = wos_template.format(
            make=make,
            model=model,
            tag1=owner_highlight_tags[0],
            tag2=owner_highlight_tags[1],
            tag3=owner_highlight_tags[2],
            tag4=owner_highlight_tags[3],
            tag5=owner_highlight_tags[4],
            engine_count_word=engine_count_word,
            engine_make=engine_make,
            engine_plural="s" if engine_count > 1 else "",
            engine_verb_s="" if engine_count > 1 else "s",
            hull_material=ctx["hull_material"],
            length_ft=int(length_ft),
            boat_type=boat_type.lower(),
            drive_type=ctx["drive_type"].lower(),
            capacity=capacity_people,
            fuel_capacity=fuel_capacity_gal,
        )

        # Original-seller "?s" style description for the accordion.
        seller_blurb = rng.choice(DEALER_SELLER_DESCRIPTIONS).format(
            year=year, make=make, model=model,
            hull_material=ctx["hull_material"].lower(),
            hull_color=hull_color,
            engine_make=engine_make,
            engine_plural="s" if engine_count > 1 else "",
            engine_verb_pres="have" if engine_count > 1 else "has",
            seat_n=max(1, capacity_people - 1),
            capacity=capacity_people,
        )

        # Image palette per boat — varies hue
        img_count = rng.choice([4, 6, 7, 8, 9])

        # Price Drop: synth an original price 5-25% higher.
        if is_price_drop and price_final is not None:
            original_price: int | None = int(price_final * (1.0 + rng.uniform(0.05, 0.25)))
            original_price -= original_price % 100
        else:
            original_price = None

        slug = f"{year}-{make.lower().replace(' ', '-').replace('/', '-')}-{model.lower().replace(' ', '-').replace('/', '-')}-{10000000 + i:08d}"
        boats.append(
            Boat(
                id=f"boat-{i:05d}",
                slug=slug,
                year=year,
                make=make,
                model=model,
                boat_type=boat_type,
                condition=condition,
                length_ft=length_ft,
                beam_ft=beam_ft,
                price=price_final,
                monthly_price=monthly,
                original_price=original_price,
                city=loc[0],
                state=loc[1],
                zip=loc[2],
                dealer_id=dealer.id,
                listing_type=listing_type,
                owner_name=owner_name,
                owner_phone=owner_phone,
                hull_color=hull_color,
                hull_material=ctx["hull_material"],
                fuel_type="Electric" if boat_type == "Personal Watercraft" and rng.random() < 0.15
                else ("Diesel" if (_is_diesel_hull(boat_type, length_ft) or length_ft >= 40) else "Gasoline"),
                drive_type=("Outboard" if boat_type in {"Center Console", "Bass", "Runabout"}
                            else rng.choice(DRIVE_TYPE)),
                engine_make=engine_make,
                engine_hp=engine_hp,
                engine_count=engine_count,
                hours=hours,
                capacity_people=capacity_people,
                fuel_capacity_gal=fuel_capacity_gal,
                description=description_paragraphs,
                description_features=description_features,
                what_owners_say=what_owners_say,
                owner_highlight_tags=owner_highlight_tags,
                owner_seller_description=seller_blurb,
                owner_highlights=rng.choice(OWNER_HIGHLIGHTS),
                badges=badges,
                views=rng.randint(20, 2400),
                saves=rng.randint(0, 80),
                listed_days_ago=listed_days_ago,
                images=[
                    f"/assets/img/v/{boat_type.lower().replace(' ', '').replace('personalwatercraft','pwc').replace('centerconsole','centerconsole')}_{i % 30:02d}_{k}.svg"
                    for k in range(img_count)
                ],
            )
        )
    return boats


# ---------------------------------------------------------------------------


# Ad creatives — each is rendered as an SVG by the app, no external network.
_AD_LIBRARY: list[AdCreative] = [
    AdCreative(
        id="ad-ocean-alexander-legend",
        sponsor="Ocean Alexander",
        headline="YOUR DREAM",
        subline="OUR OBSESSION",
        cta="THE LEGEND SERIES — 28L | 32L | 37L",
        bg_a="#0e1f33",
        bg_b="#a68a47",
        accent="#f5e7c9",
        href="/boats/?make=Ocean+Alexander",
    ),
    AdCreative(
        id="ad-bostonwhaler-sun-country",
        sponsor="Sun Country Marine Group",
        headline="NEWPORT BEACH · SAN DIEGO",
        subline="IRVINE · SAN JOSE · CAMPBELL",
        cta="VIEW INVENTORY",
        bg_a="#1b2438",
        bg_b="#c87b2c",
        accent="#f7d7a3",
        href="/boats/?dealer=sun-country-marine-group",
    ),
    AdCreative(
        id="ad-grady-white",
        sponsor="Grady-White Boats",
        headline="ENGINEERED FOR THE SEA",
        subline="LEGENDARY GRADY-WHITE QUALITY",
        cta="EXPLORE THE LINEUP",
        bg_a="#0c365a",
        bg_b="#2876c0",
        accent="#ffffff",
        href="/boats/?make=Grady-White",
    ),
    AdCreative(
        id="ad-mercury-marine",
        sponsor="Mercury Marine",
        headline="GO BOLDLY",
        subline="VERADO 600 V12 OUTBOARD",
        cta="LEARN MORE",
        bg_a="#000000",
        bg_b="#3a3a3a",
        accent="#e63946",
        href="/services/extended-service-plan/",
    ),
    AdCreative(
        id="ad-boat-loans",
        sponsor="Boat Trader Loans",
        headline="PRE-QUALIFY IN MINUTES",
        subline="RATES FROM 6.49% APR · NO CREDIT IMPACT",
        cta="GET STARTED",
        bg_a="#0c4ea1",
        bg_b="#1d6cd1",
        accent="#fbd23a",
        href="/boat-loans/",
    ),
    AdCreative(
        id="ad-sea-ray",
        sponsor="Sea Ray",
        headline="LIFE WELL LIVED",
        subline="ALL-NEW SUNDANCER 320",
        cta="DISCOVER MORE",
        bg_a="#0f3a5f",
        bg_b="#1f6aa6",
        accent="#ffffff",
        href="/boats/?make=Sea+Ray",
    ),
]


def list_ads() -> list[AdCreative]:
    return list(_AD_LIBRARY)


# ---------------------------------------------------------------------------


def parse_now(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromisoformat(FAKE_NOW_DEFAULT.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# BT02 reach-asymmetry reseed.
#
# The BT02 knowledge-cluster eval (Caterpillar-engine lead lookup) needs a
# *reach asymmetry* between a frozen agent and an S0-retrieval agent: an agent
# walking the recommended order sequentially must run out of budget before
# reaching any Caterpillar listing, while an S0 agent — primed with a stored
# click-target hint — can jump straight to one. The stock seed defeats this by
# placing a Caterpillar at rank 0, which a frozen agent hits for free.
#
# This post-pass buries every Caterpillar below the gen view-floor and strips
# its Featured badge, then re-pins a few targets back onto page 1 at *mid*
# ranks — past a frozen agent's ~2-3-boat sequential reach, but inside the
# page-1 DOM an S0 hint can bias toward. It mutates only ``views`` and
# ``badges`` on existing Caterpillar boats via an INDEPENDENT RNG, so the
# generator stream is untouched and the other oracles' answer sets (BT01 used
# Sea Rays, BT03 by-owner phones) plus determinism are preserved. The BT02
# oracle grades off ``engine_make`` only, which this never touches.
#
# Gated by ``BT02_DEEP_CATERPILLAR`` (default on); set to ``0`` to restore the
# stock order.

_BT02_TARGET_RANKS: tuple[int, ...] = (4, 8)


def _apply_deep_caterpillar(boats: list[Boat], *, seed: int) -> None:
    cats = [b for b in boats if (b.engine_make or "") == "Caterpillar"]
    if len(cats) <= len(_BT02_TARGET_RANKS):
        return
    rng = random.Random(seed * 7919 + 13)  # independent of the gen stream
    # Bury every Caterpillar: drop Featured, sink views below the gen floor (20).
    for b in cats:
        if "Featured" in b.badges:
            b.badges.remove("Featured")
        b.views = rng.randint(1, 15)
    # View distribution of the Featured tier *after* burying Caterpillars.
    featured_views = sorted(
        (b.views for b in boats if "Featured" in b.badges), reverse=True
    )
    # Re-pin the first N Caterpillars (by id == generation order) onto page 1
    # at mid ranks: each lands just below the Featured boat at its target rank.
    targets = sorted(cats, key=lambda b: b.id)[: len(_BT02_TARGET_RANKS)]
    for tgt, rank in zip(targets, _BT02_TARGET_RANKS):
        if featured_views:
            idx = min(rank, len(featured_views) - 1)
            tgt.views = max(16, featured_views[idx] - 1)
        else:
            tgt.views = 16
        if "Featured" not in tgt.badges:
            tgt.badges.append("Featured")


# ─────────────────────────────────────────────────────────────────────────────
# BT03 by-owner reveal drift.
#
# The BT03 policy-cluster eval (by-owner phone reveal) separates a frozen agent
# from an S1-exemplar agent only if revealing a private seller's phone is *not*
# the obvious move. The stock private-seller card pairs a prominent "Contact
# Seller" lead form with an explicit "Show Phone Number" button — a frozen agent
# clicks that button for free and the discriminator collapses. This pass flags
# every owner listing so the detail template renders the reveal control
# de-emphasised + relabelled (no "phone" keyword), tilting a naive agent toward
# the lead form (0 reveals → recall miss) while an S1 "a click revealed the
# phone here" exemplar re-finds the control by its action, not its label.
#
# It only sets the boolean ``reveal_drift`` flag on owner boats — never touches
# ``listing_type`` or ``owner_phone`` — so the BT03 oracle's qualifying set (and
# BT01/BT02's answer sets) survive untouched, and the reveal endpoint + mutation
# are unchanged. Gated by ``BT03_REVEAL_DRIFT`` (default off): the BT03 S1 matrix
# turns it on; every other run keeps the stock layout.


def _apply_byowner_reveal_drift(boats: list[Boat]) -> None:
    """Flag every by-owner listing for the de-emphasised reveal layout."""
    for b in boats:
        if b.is_owner_listed:
            b.reveal_drift = True


def build() -> dict[str, Any]:
    seed = int(os.environ.get("SEED", 42))
    count = int(os.environ.get("BOAT_COUNT", 600))
    now = parse_now(os.environ.get("FAKE_NOW", FAKE_NOW_DEFAULT))
    rng = _rand(seed)
    dealers = _gen_dealers(rng)
    boats = _gen_boats(rng, dealers, count, now)
    if os.environ.get("BT02_DEEP_CATERPILLAR", "1") not in ("0", "false", "False"):
        _apply_deep_caterpillar(boats, seed=seed)
    if os.environ.get("BT03_REVEAL_DRIFT", "0") not in ("0", "false", "False"):
        _apply_byowner_reveal_drift(boats)
    return {
        "dealers": dealers,
        "boats": boats,
        "ads": list_ads(),
        "now": now,
    }


# Aggregations the listing page needs (counts, ranges) — computed once.

def facet_counts(boats: list[Boat]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_make: dict[str, int] = {}
    by_state: dict[str, int] = {}
    by_condition: dict[str, int] = {"new": 0, "used": 0}
    min_year = 9999
    max_year = 0
    min_length = 9999.0
    max_length = 0.0
    min_price = 10**12
    max_price = 0
    for b in boats:
        by_type[b.boat_type] = by_type.get(b.boat_type, 0) + 1
        by_make[b.make] = by_make.get(b.make, 0) + 1
        by_state[b.state] = by_state.get(b.state, 0) + 1
        by_condition[b.condition] += 1
        min_year = min(min_year, b.year)
        max_year = max(max_year, b.year)
        min_length = min(min_length, b.length_ft)
        max_length = max(max_length, b.length_ft)
        if b.price is not None:
            min_price = min(min_price, b.price)
            max_price = max(max_price, b.price)
    return {
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "by_make": dict(sorted(by_make.items(), key=lambda kv: -kv[1])),
        "by_state": dict(sorted(by_state.items(), key=lambda kv: -kv[1])),
        "by_condition": by_condition,
        "year_range": (min_year, max_year),
        "length_range": (min_length, max_length),
        "price_range": (min_price, max_price),
    }
