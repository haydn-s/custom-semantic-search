"""Turn AllCards.json (MTGJSON v4, keyed by card name) into chunks for semantic search.

Every chunk is a dict with:
    id          stable, unique id for Chroma
    kind        "card", "ruling" or "foreign"
    document    what a person reads in search results (labeled, original wording)
    embed_text  what the model embeds (rules text first, mana symbols in words)
    metadata    flat str/int/float/bool fields for Chroma `where` filters

Chunk kinds:
    card     one per card. The document lists name, mana cost, converted mana
             cost, colors, type line, rules text and power/toughness or loyalty.
             The embedding leads with the rules text and leaves out the numeric
             fields, which retrieved noticeably better in testing; numbers such
             as converted mana cost are filterable through metadata instead.
    ruling   one per official ruling, prefixed with the card's name so it
             still makes sense when it comes back on its own
    foreign  one per translation (off by default: all-MiniLM-L6-v2 only
             understands English, so these need a multilingual model)

Run it directly to see a summary and a few sample chunks:
    python chunk_cards.py
    python chunk_cards.py --out chunks.jsonl   # write every chunk for inspection
    python chunk_cards.py --max-cards 5000     # only the 5,000 most popular cards
"""

import argparse
import json
import re
from collections import Counter

COLOR_NAMES = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}
SYMBOL_RE = re.compile(r"\{([^}]+)\}")
SYMBOL_RUN_RE = re.compile(r"(?:\{[^}]+\})+")

# Symbols in a cost that are not mana.
NON_MANA_SYMBOLS = {"T": "tap", "Q": "untap", "E": "energy", "CHAOS": "chaos"}


# ---------------------------------------------------------------------------
# Mana symbols -> words. The tokenizer splits "{T}" into "{", "t", "}", which
# carries no meaning, so "{T}: Add {G}." is embedded as "tap: Add 1 green mana."
# ---------------------------------------------------------------------------

def _mana_word(symbol):
    """Name of one mana symbol, without braces: 'U' -> 'blue', 'W/B' -> 'white-or-black'."""
    if symbol in COLOR_NAMES:
        return COLOR_NAMES[symbol]
    if symbol == "C":
        return "colorless"
    if symbol == "S":
        return "snow"
    if "/" in symbol:
        left, right = symbol.split("/", 1)
        if right == "P":
            return f"Phyrexian {_mana_word(left)}"
        left_word = f"{left}-generic" if left.isdigit() else _mana_word(left)
        return f"{left_word}-or-{_mana_word(right)}"
    return symbol


def describe_symbols(run):
    """Spell out a run of adjacent symbols: '{2}{U}{U}' -> '2 generic, 2 blue mana', '{T}' -> 'tap'."""
    generic = ""  # e.g. "2", "X", "XX"
    colored = Counter()
    other = Counter()
    for symbol in SYMBOL_RE.findall(run):
        if symbol in NON_MANA_SYMBOLS:
            other[NON_MANA_SYMBOLS[symbol]] += 1
        elif symbol.isnumeric() or symbol in ("X", "Y", "Z"):
            generic += symbol
        else:
            colored[_mana_word(symbol)] += 1

    parts = [f"{generic} generic"] if generic else []
    parts += [f"{count} {word}" for word, count in colored.items()]
    mana = ", ".join(parts) + " mana" if parts else ""
    extras = [word if count == 1 else f"{count} {word}" for word, count in other.items()]
    return ", ".join(filter(None, [mana, *extras]))


def humanize(text):
    """Replace every run of {symbols} in a string with words the embedding model understands."""
    return SYMBOL_RUN_RE.sub(lambda m: describe_symbols(m.group(0)), text)


# ---------------------------------------------------------------------------
# Card chunk text
# ---------------------------------------------------------------------------

def _number(value):
    return str(int(value)) if float(value).is_integer() else str(value)


def _color_list(codes):
    return ", ".join(COLOR_NAMES[c] for c in codes) if codes else "colorless"


def card_document(card):
    """Labeled description of one card face, mana symbols as printed. Shown to people."""
    lines = [f"Name: {card['name']}"]
    faces = card.get("names") or []
    if len(faces) > 1:
        lines.append(f"Card faces: {' // '.join(faces)}")
    if card.get("manaCost"):
        lines.append(f"Mana cost: {card['manaCost']}")
    lines.append(f"Converted mana cost: {_number(card['convertedManaCost'])}")
    lines.append(f"Colors: {_color_list(card.get('colors', []))}")
    identity = card.get("colorIdentity", [])
    if identity != card.get("colors", []):
        lines.append(f"Color identity: {_color_list(identity)}")
    lines.append(f"Type: {card['type']}")
    if card.get("text"):
        lines.append(f"Rules text: {card['text']}")
    if "power" in card:
        lines.append(f"Power/toughness: {card['power']}/{card['toughness']}")
    if "loyalty" in card:
        lines.append(f"Loyalty: {card['loyalty']}")
    if "hand" in card:
        lines.append(f"Hand modifier: {card['hand']}, life modifier: {card['life']}")
    if card.get("leadershipSkills", {}).get("commander"):
        lines.append("Can be your commander")
    return "\n".join(lines)


def card_embed_text(card):
    """What the model embeds for a card: rules text first, then type, colors, name and cost.

    Field labels and numbers are left out: every chunk shares them, so they
    pull all embeddings toward each other and drown out the rules text.
    """
    lines = []
    if card.get("text"):
        lines.append(card["text"])
    lines.append(f"{card['type']}, {_color_list(card.get('colors', []))}")
    lines.append(f"{card['name']}, costs {card['manaCost']}" if card.get("manaCost")
                 else card["name"])
    if "power" in card:
        lines.append(f"{card['power']}/{card['toughness']}")
    if "loyalty" in card:
        lines.append(f"loyalty {card['loyalty']}")
    return humanize("\n".join(lines))


def filter_metadata(card):
    """Fields to filter on, e.g. where={'color_R': True}. Every chunk of a card carries these."""
    colors = card.get("colors", [])
    meta = {"name": card["name"], "converted_mana_cost": float(card["convertedManaCost"])}
    for code in COLOR_NAMES:
        meta[f"color_{code}"] = code in colors
    for fmt, status in card.get("legalities", {}).items():
        meta[f"legality_{fmt}"] = status  # "Legal", "Restricted" or "Banned"
    return meta


def card_metadata(card):
    """Filter fields plus what a results page needs to show the card.

    Rulings carry only the filter fields: they are 64% of all chunks, and
    every extra field costs memory in the deployed app.
    """
    meta = {
        **filter_metadata(card),
        "mana_cost": card.get("manaCost", ""),
        "colors": ",".join(card.get("colors", [])),
        "color_identity": ",".join(card.get("colorIdentity", [])),
        "type_line": card["type"],
        "types": ",".join(card["types"]),
        "layout": card["layout"],
        "text": card.get("text", ""),
    }
    if "power" in card:
        meta["power"] = card["power"]
        meta["toughness"] = card["toughness"]
    if "loyalty" in card:
        meta["loyalty"] = card["loyalty"]
    if "edhrecRank" in card:
        meta["edhrec_rank"] = card["edhrecRank"]
    return meta


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

def _chunk(chunk_id, kind, document, embed_text, metadata):
    return {
        "id": chunk_id,
        "kind": kind,
        "document": document,
        "embed_text": embed_text,
        "metadata": {**metadata, "kind": kind},
    }


def card_chunks(card, include_foreign=False):
    meta = filter_metadata(card)
    uuid = card["uuid"]
    name = card["name"]

    chunks = [_chunk(uuid, "card", card_document(card), card_embed_text(card), card_metadata(card))]

    for i, ruling in enumerate(card.get("rulings", [])):
        chunks.append(_chunk(
            f"{uuid}-ruling-{i}", "ruling",
            document=f"Ruling for {name} ({ruling['date']}): {ruling['text']}",
            embed_text=humanize(f"Ruling for {name}: {ruling['text']}"),
            metadata={**meta, "ruling_date": ruling["date"]},
        ))

    if include_foreign:
        for entry in card.get("foreignData", []):
            lines = [f"Name: {entry['name']}", f"English name: {name}"]
            if entry.get("type"):
                lines.append(f"Type: {entry['type']}")
            if entry.get("text"):
                lines.append(f"Rules text: {entry['text']}")
            if entry.get("flavorText"):
                lines.append(f"Flavor text: {entry['flavorText']}")
            document = "\n".join(lines)
            language = entry["language"]
            slug = re.sub(r"\W+", "-", language.lower()).strip("-")
            chunks.append(_chunk(f"{uuid}-{slug}", "foreign", document, humanize(document),
                                 {**meta, "language": language}))

    return chunks


def select_cards(cards, max_cards=None):
    """The `max_cards` most-played cards (lowest EDHREC rank), or all of them if max_cards is None.

    The file is alphabetical, so its first N entries would only cover names
    starting with the first few letters; ranking by popularity keeps the cards
    people actually search for. The result stays in file order.
    """
    if max_cards is None:
        return cards
    ranked = sorted(cards, key=lambda c: c.get("edhrecRank", float("inf")))
    keep = {c["uuid"] for c in ranked[:max_cards]}
    return [c for c in cards if c["uuid"] in keep]


def build_chunks(path="AllCards.json", include_foreign=False, max_cards=None):
    """Load the card file and return the chunks for every card (or the `max_cards` most popular)."""
    with open(path, encoding="utf-8") as f:
        cards = list(json.load(f).values())
    return [chunk for card in select_cards(cards, max_cards)
            for chunk in card_chunks(card, include_foreign)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cards", default="AllCards.json", help="path to AllCards.json")
    parser.add_argument("--out", help="write all chunks to this .jsonl file")
    parser.add_argument("--foreign", action="store_true", help="include translated chunks")
    parser.add_argument("--max-cards", type=int, help="only the N most popular cards (by EDHREC rank)")
    args = parser.parse_args()

    chunks = build_chunks(args.cards, include_foreign=args.foreign, max_cards=args.max_cards)
    print(f"{len(chunks):,} chunks:", dict(Counter(c["kind"] for c in chunks)))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")
    else:
        samples = {"Llanowar Elves", "Fire", "Grim Lavamancer"}
        for chunk in chunks:
            if chunk["metadata"]["name"] in samples:
                print(f"\n=== {chunk['kind']}: {chunk['id']} ===")
                print("--- document ---\n" + chunk["document"])
                print("--- embed_text ---\n" + chunk["embed_text"])


if __name__ == "__main__":
    main()
