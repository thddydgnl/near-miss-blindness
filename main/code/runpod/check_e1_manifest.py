"""Mac-side check of the E1 phase-2 manifest (standard library only): every imagine job's dep is a cf job of the
manifest and both fall in the same shard for N pods; per-shard and per-task counts.
Usage: python3 check_e1_manifest.py <manifest.jsonl> <N>"""
import collections
import json
import sys
import zlib

jobs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = int(sys.argv[2])
cf = {j["job_id"]: j for j in jobs if j["kind"] == "cf"}
im = [j for j in jobs if j["kind"] == "imagine"]
key = lambda j: zlib.crc32(j.get("dep", j["job_id"]).encode()) % n
assert all(j["dep"] in cf for j in im), "imagine without cf"
assert all(key(j) == key(cf[j["dep"]]) for j in im), "dep in another shard"
assert len({j["job_id"] for j in jobs}) == len(jobs), "duplicate ids"
print("jobs", len(jobs), "cf", len(cf), "imagine", len(im))
print("per shard", sorted(collections.Counter(key(j) for j in im).items()))
print("per task", sorted(collections.Counter(cf[j["dep"]]["task"] for j in im).items()))
