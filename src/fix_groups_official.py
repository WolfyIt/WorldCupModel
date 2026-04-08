"""
fix_groups_official.py
======================
Post-processing script that injects the official FIFA 2026 draw results
directly into the notebook JSON, replacing provisional placeholder groups.

Run from the repo root:
    python src/fix_groups_official.py

The notebook (mundial_2026_v2_advanced.ipynb) must be at the repo root.
"""

import json
from pathlib import Path

# Resolve notebook path relative to this script's parent directory (repo root)
NB = Path(__file__).resolve().parent.parent / "mundial_2026_v2_advanced.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

changed = 0

for cell in nb["cells"]:
    src = "".join(cell["source"])

    # Fix 1: Replace TBD placeholder keys in initial_ratings_v2
    if "TBD_K1" in src and "initial_ratings_v2" in src and "TBD_K2" in src:
        old = (
            "    # Spots TBD (repesca / clasificaciones pendientes)\n"
            "    'TBD_K1': 1480, 'TBD_K2': 1480,\n"
            "    'TBD_L1': 1480, 'TBD_L2': 1480, 'TBD_L3': 1480, 'TBD_L4': 1480,"
        )
        new = (
            "    # Playoff / pending qualification spots\n"
            "    # A4=DEN/MKD/CZE/IRL  B4=ITA/NIR/WAL/BIH  D4=TUR/ROU/SVK/KOS\n"
            "    # F4=UKR/SWE/POL/ALB  I1=BOL/SUR/IRQ       K1=NCL/JAM/COD\n"
            "    'TBD_A4': 1490, 'TBD_B4': 1480,\n"
            "    'TBD_D4': 1475, 'TBD_F4': 1472,\n"
            "    'TBD_I1': 1468, 'TBD_K1': 1462,"
        )
        if old in src:
            src = src.replace(old, new)
            changed += 1
            print("Fixed: initial_ratings_v2 TBD keys")
        else:
            print("MISS TBD fix — searching context...")
            idx = src.find("TBD_K1")
            print(repr(src[idx - 100 : idx + 200]))

    # Fix 2: Replace provisional GROUPS_2026_V2 with official FIFA 2026 draw
    if "GROUPS_2026_V2" in src and "'A':" in src:
        old_block_start = "# ---- Grupos FIFA 2026"
        idx_start = src.find(old_block_start)
        if idx_start == -1:
            print("MISS: groups block start not found")
        else:
            idx_end = src.find("def _predict_match_v2")
            if idx_end == -1:
                print("MISS: groups block end not found")
            else:
                # Official FIFA WC 2026 draw — updated March 2026
                new_block = '''\
# ---- FIFA 2026 Groups — OFFICIAL DRAW ----
# Source: official FIFA WC2026 draw ceremony
GROUPS_2026_V2 = {
    'A': ['Mexico',      'South Africa', 'South Korea',  'TBD_A4'],    # TBD_A4 = DEN/MKD/CZE/IRL
    'B': ['Canada',      'Qatar',        'Switzerland',  'TBD_B4'],    # TBD_B4 = ITA/NIR/WAL/BIH
    'C': ['Brazil',      'Morocco',      'Haiti',        'Scotland'],
    'D': ['USA',         'Paraguay',     'Australia',    'TBD_D4'],    # TBD_D4 = TUR/ROU/SVK/KOS
    'E': ['Germany',     'Curazao',      'Ivory Coast',  'Ecuador'],
    'F': ['Netherlands', 'Japan',        'Tunisia',      'TBD_F4'],    # TBD_F4 = UKR/SWE/POL/ALB
    'G': ['Belgium',     'Egypt',        'Iran',         'New Zealand'],
    'H': ['Spain',       'Cape Verde',   'Saudi Arabia', 'Uruguay'],
    'I': ['TBD_I1',      'France',       'Senegal',      'Norway'],    # TBD_I1 = BOL/SUR/IRQ
    'J': ['Argentina',   'Algeria',      'Austria',      'Jordan'],
    'K': ['TBD_K1',      'Portugal',     'Uzbekistan',   'Colombia'],  # TBD_K1 = NCL/JAM/COD
    'L': ['England',     'Croatia',      'Ghana',        'Panama'],
}
# Normalize team names to match internal format
GROUPS_2026_V2 = {
    g: [normalize_team(t) for t in teams_]
    for g, teams_ in GROUPS_2026_V2.items()
}

'''
                src = src[:idx_start] + new_block + src[idx_end:]
                changed += 1
                print("Fixed: GROUPS_2026_V2")

    cell["source"] = src.splitlines(keepends=True)

print(f"\nTotal changes applied: {changed}")
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Saved: {NB}")
