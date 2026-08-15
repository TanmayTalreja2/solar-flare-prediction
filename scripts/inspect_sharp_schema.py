import drms


SERIES_NAME = "hmi.sharp_cea_720s"


def main() -> None:
    """Inspect the JSOC SHARP CEA series schema."""

    client = drms.Client()

    print(f"Inspecting: {SERIES_NAME}")
    print()

    series_info = client.info(SERIES_NAME)

    print("========== PRIME KEYS ==========")
    print(series_info.primekeys)
    print()

    print("========== SERIES NOTE ==========")
    print(series_info.note)
    print()

    print("========== KEYWORD COUNT ==========")
    print(len(series_info.keywords))
    print()

    print("========== RELEVANT KEYWORDS ==========")

    keywords = series_info.keywords

    relevant_terms = (
        "quality",
        "conf",
        "error",
        "err",
        "bitmap",
        "disambig",
        "mask",
    )

    for keyword in keywords.index:
        keyword_lower = keyword.lower()

        if any(
            term in keyword_lower
            for term in relevant_terms
        ):
            info = keywords.loc[keyword]

            print(f"\n{keyword}")
            print(f"  Type: {info['type']}")
            print(f"  Units: {info['units']}")
            print(f"  Default: {info['defval']}")
            print(f"  Note: {info['note']}")

    print()
    print("========== SEGMENTS ==========")

    print(
        list(series_info.segments.index)
    )


if __name__ == "__main__":
    main()