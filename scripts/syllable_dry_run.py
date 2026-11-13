#!/usr/bin/env python3
"""PROP-8 dry run, Python half (idtap-contract 0.3.0, rollout step 3).

For every piece JSON document in a directory, run two serializations side by
side and compare their legacy phonetic fields:

  A  today's released client (this repo's origin/main): load -> save
  B  this branch (contract 0.3.0):                      load -> heal -> save
     (the healed tier is adopted, so the save dual-writes: the legacy fields
     are the projection of the syllables)

GATE: on every piece, B's legacy fields equal A's byte-for-byte: the nine wire
fields (vowel, startConsonant{,Hindi,Ipa,EngTrans}, endConsonant{,...}) and the
'0.00' / '1.00' consonant articulations of every row-0 trajectory. Zero diffs.

Also reported, not gated: A against the raw stored bytes per field (what
today's load -> save already normalizes), counts of '' and null in
vowel/startConsonant/endConsonant, tracks whose instrumentation is not vocal
but carry phonetic fields, the invariant violations B finds, and load/save
failures. B is also saved a second way, with the tier flag off (no `system`, so
nothing new is written), and that output is compared against A too.

A and B run in separate Python processes: A imports `idtap` from a
`git archive` of the baseline ref, B from this checkout. Trajectories are
matched by position (track, phrase, row-0 index), not uniqueId, because a
trajectory stored without a uniqueId gets a fresh one in each process.

Usage:
  uv run python scripts/syllable_dry_run.py DIR [--baseline-ref origin/main]
                                                [--report report.json]
DIR holds *.json files (one document, or a JSON list of documents, per file)
and/or *.jsonl files (one document per line). Mongo extended JSON
({"$oid": ...}, {"$date": ...}) is accepted. Exit status 0 = gate passed.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import io

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIRE_FIELDS = ["vowel",
               "startConsonant", "startConsonantHindi", "startConsonantIpa", "startConsonantEngTrans",
               "endConsonant", "endConsonantHindi", "endConsonantIpa", "endConsonantEngTrans"]
PHONETIC = ("vowel", "startConsonant", "endConsonant")
VOCAL = ("Vocal (M)", "Vocal (F)")


# --------------------------------------------------------------------------- input
def _unwrap(x):
    """Mongo extended JSON -> plain values ({"$oid"} -> str; {"$date"} kept,
    Piece parses it). Applied identically for A and B."""
    if isinstance(x, dict):
        if set(x) == {"$oid"}:
            return x["$oid"]
        return {k: _unwrap(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_unwrap(v) for v in x]
    return x


AS_LEGACY = False  # set by --as-legacy (passed on to the workers)


def _as_legacy(doc):
    """Drop the 0.3.0 keys: what a pre-0.3.0 database holds."""
    doc = {k: v for k, v in doc.items() if k not in ("system", "lyricsText")}
    for track in doc.get("phraseGrid") or []:
        for ph in track or []:
            if isinstance(ph, dict):
                ph.pop("syllables", None)
    return doc


def iter_documents(src):
    """Yield (label, document) for every piece in `src`, in a stable order."""
    for label, doc in _iter_documents(src):
        yield label, (_as_legacy(doc) if AS_LEGACY else doc)


def _iter_documents(src):
    paths = sorted(glob.glob(os.path.join(src, "**", "*.json"), recursive=True)
                   + glob.glob(os.path.join(src, "**", "*.jsonl"), recursive=True))
    for path in paths:
        rel = os.path.relpath(path, src)
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for n, line in enumerate(f):
                    if line.strip():
                        yield f"{rel}:{n + 1}", _unwrap(json.loads(line))
                continue
            data = json.load(f)
        if isinstance(data, list):
            for n, doc in enumerate(data):
                yield f"{rel}[{n}]", _unwrap(doc)
        elif isinstance(data, dict) and "pieceJson" in data and "phraseGrid" not in data:
            # an idtap-contract fixture: the piece is under pieceJson (+ cases)
            yield rel, _unwrap(data["pieceJson"])
        elif isinstance(data, dict) and "cases" in data:
            for c in data["cases"]:
                yield f"{rel}#{c.get('name')}", _unwrap(c["pieceJson"])
        elif isinstance(data, dict) and "phraseGrid" in data:
            yield rel, _unwrap(data)


# --------------------------------------------------------------------------- extraction
def _dump(v):
    return json.dumps(v, ensure_ascii=False)


def legacy_grid(doc):
    """[[ [ {field: json text} per row-0 traj ] per phrase ] per track]: the
    legacy fields exactly as serialized (absent fields omitted)."""
    out = []
    for track in doc.get("phraseGrid") or []:
        trow = []
        for ph in track or []:
            grid = (ph or {}).get("trajectoryGrid") or [[]]
            prow = []
            for t in (grid[0] if grid else []) or []:
                rec = {k: _dump(t[k]) for k in WIRE_FIELDS if k in t}
                arts = t.get("articulations") or {}
                for key, name in (("0.00", "art0"), ("1.00", "art1")):
                    a = arts.get(key)
                    if isinstance(a, dict) and a.get("name") == "consonant":
                        rec[name] = _dump(a)
                prow.append(rec)
            trow.append(prow)
        out.append(trow)
    return out


def uid_grid(doc):
    return [[[t.get("uniqueId") for t in (((ph or {}).get("trajectoryGrid") or [[]])[0] or [])]
             for ph in track or []] for track in doc.get("phraseGrid") or []]


def raw_stats(doc):
    """Counts over the stored document (not gated)."""
    empty = collections.Counter()
    nulls = collections.Counter()
    nonvocal = []
    instr = doc.get("instrumentation") or []
    for k, track in enumerate(doc.get("phraseGrid") or []):
        phonetic = 0
        for ph in track or []:
            for t in (((ph or {}).get("trajectoryGrid") or [[]])[0] or []):
                for f in PHONETIC:
                    if f in t and t[f] is None:
                        nulls[f] += 1
                    elif t.get(f) == "":
                        empty[f] += 1
                if any(t.get(f) is not None for f in PHONETIC):
                    phonetic += 1
        label = instr[k] if k < len(instr) else None
        if phonetic and label not in VOCAL:
            nonvocal.append({"track": k, "instrumentation": label, "phoneticTrajs": phonetic})
    return {"empty": dict(empty), "null": dict(nulls), "nonVocalPhonetic": nonvocal,
            "hasSystem": "system" in doc,
            "storedSyllablePhrases": sum(1 for tr in doc.get("phraseGrid") or []
                                         for ph in tr or [] if isinstance(ph, dict)
                                         and ph.get("syllables") is not None)}


# --------------------------------------------------------------------------- workers
def worker(mode, src, out_path, idtap_root):
    """Run in a subprocess: import idtap from `idtap_root`, serialize every
    document, write one JSON line per document."""
    sys.path.insert(0, idtap_root)
    import warnings
    import idtap
    from idtap.classes.piece import Piece
    where = os.path.realpath(os.path.dirname(idtap.__file__))
    if not where.startswith(os.path.realpath(idtap_root) + os.sep):
        raise SystemExit(f"worker {mode}: imported idtap from {where}, not {idtap_root}")
    with open(out_path, "w", encoding="utf-8") as out:
        for label, doc in iter_documents(src):
            rec = {"label": label, "id": doc.get("_id")}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    piece = Piece.from_json(json.loads(json.dumps(doc)))
                    if mode == "B":
                        flag_off = piece.to_json()
                        rec["legacyFlagOff"] = legacy_grid(flag_off)
                        piece = Piece.from_json(json.loads(json.dumps(doc)))
                        rec["violations"] = [list(v) for v in piece.syllable_violations()]
                        piece.adopt_healed_syllables()
                    saved = piece.to_json()
                rec["legacy"] = legacy_grid(saved)
                rec["uids"] = uid_grid(saved)
                if mode == "A":
                    rec["raw"] = legacy_grid(doc)
                    rec["rawUids"] = uid_grid(doc)
                    rec["stats"] = raw_stats(doc)
                rec["ok"] = True
            except Exception as e:  # reported, never hidden
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"worker {mode}: idtap {idtap.__version__} from {where}", file=sys.stderr)


def export_baseline(ref, dest):
    """`git archive REF idtap` unpacked into dest (the released client's code)."""
    data = subprocess.run(["git", "-C", REPO, "archive", ref, "idtap"],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(dest, filter="data")
        else:  # pragma: no cover - Python < 3.10.12
            tf.extractall(dest)
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", ref],
                         check=True, capture_output=True, text=True).stdout.strip()
    return sha


def run_worker(mode, src, out_path, root):
    # cwd outside both trees, so neither checkout's idtap is importable by accident
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", mode,
           "--idtap-root", root, "--out", out_path, src]
    if AS_LEGACY:
        cmd.append("--as-legacy")
    subprocess.run(cmd, check=True, cwd=os.path.dirname(out_path))


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- compare
def compare_grids(ga, gb):
    """-> list of (track, phrase, traj, field, a, b) where they differ, and a
    structure note when the grids do not have the same shape."""
    diffs = []
    if [[len(p) for p in t] for t in ga] != [[len(p) for p in t] for t in gb]:
        return diffs, "row-0 trajectory layout differs"
    for k, (ta, tb) in enumerate(zip(ga, gb)):
        for pi, (pa, pb) in enumerate(zip(ta, tb)):
            for i, (ra, rb) in enumerate(zip(pa, pb)):
                for f in sorted(set(ra) | set(rb)):
                    if ra.get(f) != rb.get(f):
                        diffs.append((k, pi, i, f, ra.get(f), rb.get(f)))
    return diffs, None


def compare_raw(raw, raw_uids, saved, saved_uids):
    """A vs the stored bytes, matched by uniqueId (today's load may add or
    merge trajectories, so positions can shift). -> Counter of field changes."""
    by_uid = {}
    for t_l, t_u in zip(saved, saved_uids):
        for p_l, p_u in zip(t_l, t_u):
            for r, u in zip(p_l, p_u):
                if u:
                    by_uid[u] = r
    changed = collections.Counter()
    unmatched = 0
    for t_l, t_u in zip(raw, raw_uids):
        for p_l, p_u in zip(t_l, t_u):
            for r, u in zip(p_l, p_u):
                if not u or u not in by_uid:
                    unmatched += 1
                    continue
                s = by_uid[u]
                for f in set(r) | set(s):
                    if r.get(f) != s.get(f):
                        changed[f] += 1
    return changed, unmatched


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="directory of piece JSON documents")
    ap.add_argument("--baseline-ref", default="origin/main",
                    help="git ref of the released client for A (default origin/main)")
    ap.add_argument("--report", help="write the full JSON report here")
    ap.add_argument("--max-examples", type=int, default=20)
    ap.add_argument("--as-legacy", action="store_true",
                    help="strip syllables/system/lyricsText from every input first "
                         "(simulates a pre-0.3.0 database; for testing on fixtures)")
    ap.add_argument("--tmp", help="directory for the baseline export and worker output "
                                  "(default: the system temp dir)")
    ap.add_argument("--worker", choices=["A", "B"], help=argparse.SUPPRESS)
    ap.add_argument("--idtap-root", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    src = os.path.abspath(args.src)
    global AS_LEGACY
    AS_LEGACY = args.as_legacy
    if args.worker:
        worker(args.worker, src, args.out, args.idtap_root)
        return 0

    with tempfile.TemporaryDirectory(prefix="syllable-dry-run-", dir=args.tmp) as tmp:
        base = os.path.join(tmp, "baseline")
        os.makedirs(base)
        sha = export_baseline(args.baseline_ref, base)
        a_out, b_out = os.path.join(tmp, "a.jsonl"), os.path.join(tmp, "b.jsonl")
        run_worker("A", src, a_out, base)
        run_worker("B", src, b_out, REPO)
        A, B = read_lines(a_out), read_lines(b_out)

    assert [r["label"] for r in A] == [r["label"] for r in B]
    report = {"baselineRef": args.baseline_ref, "baselineSha": sha, "pieces": len(A),
              "asLegacy": AS_LEGACY,
              "gate": {"diffPieces": 0, "diffFields": 0, "bFailedWhereAOk": 0,
                       "structure": 0, "examples": []},
              "flagOff": {"diffPieces": 0, "examples": []},
              "notGated": {"aFailed": [], "bothFailed": [], "aVsRaw": collections.Counter(),
                           "aVsRawUnmatchedTrajs": 0, "empty": collections.Counter(),
                           "null": collections.Counter(), "nonVocalPhonetic": [],
                           "violations": collections.Counter(), "violationPieces": [],
                           "piecesWithSystem": 0, "piecesWithStoredSyllables": 0}}
    g, ng, fo = report["gate"], report["notGated"], report["flagOff"]
    for a, b in zip(A, B):
        who = {"label": a["label"], "id": a.get("id")}
        if not a["ok"] and not b["ok"]:
            ng["bothFailed"].append({**who, "a": a["error"], "b": b["error"]})
            continue
        if not a["ok"]:
            ng["aFailed"].append({**who, "a": a["error"]})
            continue
        if not b["ok"]:
            g["bFailedWhereAOk"] += 1
            g["examples"].append({**who, "bError": b["error"]})
            continue
        diffs, struct = compare_grids(a["legacy"], b["legacy"])
        if struct:
            g["structure"] += 1
            g["examples"].append({**who, "structure": struct})
        if diffs:
            g["diffPieces"] += 1
            g["diffFields"] += len(diffs)
            if len(g["examples"]) < args.max_examples:
                g["examples"].append({**who, "diffs": [
                    dict(zip(("track", "phrase", "traj", "field", "a", "b"), d)) for d in diffs[:10]]})
        fdiffs, fstruct = compare_grids(a["legacy"], b["legacyFlagOff"])
        if fdiffs or fstruct:
            fo["diffPieces"] += 1
            if len(fo["examples"]) < args.max_examples:
                fo["examples"].append({**who, "structure": fstruct, "diffs": [
                    dict(zip(("track", "phrase", "traj", "field", "a", "b"), d)) for d in fdiffs[:10]]})
        changed, unmatched = compare_raw(a["raw"], a["rawUids"], a["legacy"], a["uids"])
        ng["aVsRaw"].update(changed)
        ng["aVsRawUnmatchedTrajs"] += unmatched
        st = a["stats"]
        ng["empty"].update(st["empty"])
        ng["null"].update(st["null"])
        if st["nonVocalPhonetic"]:
            ng["nonVocalPhonetic"].append({**who, "tracks": st["nonVocalPhonetic"]})
        ng["piecesWithSystem"] += st["hasSystem"]
        ng["piecesWithStoredSyllables"] += bool(st["storedSyllablePhrases"])
        if b["violations"]:
            codes = collections.Counter(c for c, _ in b["violations"])
            ng["violations"].update(codes)
            ng["violationPieces"].append({**who, "codes": dict(codes),
                                          "first": b["violations"][:5]})

    passed = (g["diffPieces"] == 0 and g["bFailedWhereAOk"] == 0 and g["structure"] == 0)
    report["gatePassed"] = passed
    for k in ("aVsRaw", "empty", "null", "violations"):
        ng[k] = dict(ng[k])

    print(f"dry run over {report['pieces']} piece(s) from {src}")
    print(f"  A = {args.baseline_ref} ({sha}) load -> save;  B = this checkout load -> heal -> save")
    print(f"GATE (B legacy fields == A, byte-for-byte): {'PASS' if passed else 'FAIL'}")
    print(f"  pieces with diffs: {g['diffPieces']} ({g['diffFields']} fields); "
          f"layout differs: {g['structure']}; B failed where A loaded: {g['bFailedWhereAOk']}")
    print(f"flag-off save (tier not written) vs A: {fo['diffPieces']} piece(s) differ")
    print("not gated:")
    print(f"  A failed to load/save: {len(ng['aFailed'])}; both failed: {len(ng['bothFailed'])}")
    print(f"  A vs raw stored, trajectories changed per field: {ng['aVsRaw'] or 'none'}"
          f" (unmatched trajs: {ng['aVsRawUnmatchedTrajs']})")
    print(f"  '' values: {ng['empty'] or 'none'};  null values: {ng['null'] or 'none'}")
    print(f"  non-vocal tracks with phonetic fields: {len(ng['nonVocalPhonetic'])} piece(s)")
    print(f"  invariant violations in B: {ng['violations'] or 'none'} "
          f"in {len(ng['violationPieces'])} piece(s)")
    print(f"  pieces with system: {ng['piecesWithSystem']}; with stored syllables: "
          f"{ng['piecesWithStoredSyllables']}")
    if g["examples"]:
        print("first gate failures:")
        for ex in g["examples"][:5]:
            print("  ", json.dumps(ex, ensure_ascii=False)[:400])
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"report: {os.path.abspath(args.report)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
