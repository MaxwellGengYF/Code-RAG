"""Re-run the human verification of eval_gold_extended.json.

The plan requires extended gold queries to be human-verified. Verification means:
each gold page's corpus text must contain the specific content the question asks
about -- otherwise an LLM wrote a plausible question whose answer is not actually
on the page it was attributed to, which would silently corrupt the eval set.

This script re-runs the needle checks recorded in eval_gold_verification.md, plus
prints the surrounding text for any page whose needles are absent so a human can
judge it (queries 4, 6, 9, 14 were verified by reading rather than needles).

Usage: uv run python eval_gold_verify.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.store import CorpusStore

#: distinctive content needles each gold page must contain. Queries absent here
#: (4, 6, 9, 14) were verified by reading the page text; they are printed with
#: context so a human can re-judge them.
NEEDLES = {
    0: ["drag", "target"],
    1: ["AndroidJavaProxy"],
    2: ["resampling", "270"],
    3: ["RequestUserAuthorization", "AsyncOperation"],
    5: ["Button", "icon"],
    7: ["AvatarIKGoal", "LeftFoot"],
    8: ["ClearRequestedMipmapLevel"],
    10: ["Compile flags", "Xcode"],
    11: ["SetOverrideSampleSettings"],
    12: ["AddLink", "NavMeshLinkData"],
    13: ["asmdef", "edit"],
    15: ["stackTraceLogType", "obsolete"],
}
#: verified by reading; show context around these terms
READ_VERIFIED = {4: "Collider Update Mode", 6: "ChangeCoordinatesTo",
                 9: "MenuItem", 14: "If null (default)"}


def main():
    store = CorpusStore(Path("corpus"))
    queries = json.loads(Path("eval_gold_extended.json").read_text(encoding="utf-8"))
    n_verified = n_read = n_fail = n_missing = 0

    for i, entry in enumerate(queries):
        rel = entry["gold"][0]
        page = store.load(rel)
        if page is None:
            print(f"{i:>2}. MISSING FROM CORPUS: {rel}  (regenerate: "
                  f"rag.py compile --only {rel} --provider <cfg>)")
            n_missing += 1
            continue
        blob = " ".join(c["text"] for c in page.get("chunks") or [])
        low = blob.lower()

        if i in READ_VERIFIED:
            needle = READ_VERIFIED[i]
            pos = low.find(needle.lower())
            if pos >= 0:
                n_read += 1
                print(f"{i:>2}. READ-VERIFIED  {entry['query'][:62]}")
                print(f"     gold: {rel}")
                print(f"     …{blob[max(0, pos - 90):pos + 150]}…")
            else:
                n_fail += 1
                print(f"{i:>2}. NEEDS RE-REVIEW  {entry['query'][:62]}")
                print(f"     gold: {rel} -- {needle!r} no longer present")
            continue

        needles = NEEDLES.get(i)
        if needles is None:
            print(f"{i:>2}. NO NEEDLES DEFINED  {entry['query'][:62]}")
            n_fail += 1
            continue
        hits = [n for n in needles if n.lower() in low]
        if len(hits) == len(needles):
            n_verified += 1
            print(f"{i:>2}. VERIFIED {len(hits)}/{len(needles)}  "
                  f"{entry['query'][:56]}")
        else:
            n_fail += 1
            print(f"{i:>2}. FAILED {len(hits)}/{len(needles)}  "
                  f"{entry['query'][:56]}")
            print(f"     gold: {rel}")
            print(f"     present: {hits}  missing: "
                  f"{[n for n in needles if n not in hits]}")

    print(f"\n== verification summary ==")
    print(f"  needle-verified : {n_verified}")
    print(f"  read-verified   : {n_read}")
    print(f"  failed          : {n_fail}")
    print(f"  missing corpus  : {n_missing}")
    print(f"  total           : {len(queries)}")
    ok = n_fail == 0 and n_missing == 0
    print(f"  VERDICT         : {'PASS' if ok else 'FAIL — see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
