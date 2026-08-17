import json
from pathlib import Path

REPO = Path("~/Gold Datasets/BASIL-main").expanduser()

# --- find the two trees ---
ann_files = sorted(REPO.rglob("annotations/**/*.json"))
art_files = sorted(REPO.rglob("articles/**/*.json"))
print(f"annotation files: {len(ann_files)}")
print(f"article files:    {len(art_files)}")

# --- an ARTICLE file: where are the sentences? ---
if art_files:
    a = json.loads(art_files[0].read_text())
    print("\nARTICLE top-level keys:", list(a.keys()))
    for k in ("body-paragraphs", "body", "sentences", "paragraphs"):
        if k in a:
            v = a[k]
            print(f"  '{k}': type={type(v)}")
            if isinstance(v, list) and v:
                print("   first element:", json.dumps(v[0], indent=2)[:500])
            break
    print("uuid:", a.get("uuid"), "| triplet-uuid:", a.get("triplet-uuid"))

# --- an ANNOTATION file: one phrase-level bias span ---
b = json.loads(ann_files[0].read_text())
pla = b.get("phrase-level-annotations", [])
print(f"\nphrase-level-annotations: {len(pla)} entries")
if pla:
    print("ONE span:", json.dumps(pla[0], indent=2))
print("uuid:", b.get("uuid"), "| triplet-uuid:", b.get("triplet-uuid"))
