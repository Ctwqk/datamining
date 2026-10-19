"""'People you may know': recommend friends by number of mutual friends (coursework HW3).

Input: one line per user, ``user<TAB>friend1,friend2,...`` (the classic adjacency-list
format of the CS246 friend-recommendation assignment). For every user the program
recommends up to ``--top`` people who are *not* yet friends, ranked by the number of
mutual friends (ties broken by the smaller user id).

Two interchangeable engines compute the same thing:

* ``spark``  - the original MapReduce-style PySpark job (needs a JDK and ``pyspark``);
* ``python`` - a plain-Python reference implementation (used as the test oracle).

Compared with the original Colab notebook export this version

* reads a local file instead of downloading ``friends.txt`` from a private Google Drive
  (the real assignment data is not in the repository; ``sample_friends.txt`` is a small
  synthetic graph, see ``make_sample_graph.py``);
* fixes a bug: ``subtractByKey`` was given an RDD keyed by ``user`` instead of the
  ``(user, friend)`` pair, so it removed nothing and existing friends were recommended;
* sorts each user's candidates itself instead of relying on ``groupByKey`` preserving
  the order of an earlier global sort (Spark does not guarantee that);
* drops the notebook-only code (``pip install``, Drive authentication, hard-coded
  ``/content/drive`` output path) and the "cross product" experiment that its author
  marked as too slow and that did not exclude existing friends either.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from collections.abc import Iterable
from itertools import permutations
from operator import add
from pathlib import Path

HERE = Path(__file__).resolve().parent
Adjacency = dict[int, list[int]]
Recommendations = dict[int, list[int]]


def parse_line(line: str) -> tuple[int, list[int]] | None:
    """``'3\\t1,2,5'`` -> ``(3, [1, 2, 5])``; lines without a tab separator are skipped."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2 or not parts[0].strip():
        return None
    friends = [int(x) for x in parts[1].split(",") if x.strip()]
    return int(parts[0]), friends


def read_adjacency(path: str | Path) -> Adjacency:
    adjacency: Adjacency = {}
    with open(path, encoding="utf8") as fh:
        for line in fh:
            parsed = parse_line(line)
            if parsed is not None:
                adjacency[parsed[0]] = parsed[1]
    return adjacency


def rank_candidates(counts: Iterable[tuple[int, int]], top_k: int) -> list[int]:
    """Candidate ids ordered by mutual-friend count (desc) then id (asc), truncated."""
    return [c for c, _ in sorted(counts, key=lambda kv: (-kv[1], kv[0]))[:top_k]]


def recommend_reference(
    adjacency: Adjacency, top_k: int = 10, users: Iterable[int] | None = None
) -> Recommendations:
    """Pure-Python implementation of the same definition as the Spark job.

    A user ``x`` is a mutual friend of the ordered pair ``(a, b)`` when ``a`` and ``b``
    both appear in ``x``'s friend list. Pairs that are already direct friends in ``a``'s
    own list are dropped. Users with no candidate do not appear in the result.
    """
    wanted = None if users is None else set(users)
    counts: dict[int, Counter] = {}
    for friends in adjacency.values():
        for a, b in permutations(dict.fromkeys(friends), 2):
            if wanted is None or a in wanted:
                counts.setdefault(a, Counter())[b] += 1
    recs: Recommendations = {}
    for a, counter in counts.items():
        already = set(adjacency.get(a, ()))
        candidates = [(b, n) for b, n in counter.items() if b not in already]
        if candidates:
            recs[a] = rank_candidates(candidates, top_k)
    return recs


def get_spark_context(master: str = "local[2]"):
    """Local SparkContext; workers must run the same Python as the driver."""
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master(master)
        .appName("friend-recommendation")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark.sparkContext


def recommend_spark(
    sc, path: str | Path, top_k: int = 10, users: Iterable[int] | None = None
) -> Recommendations:
    """MapReduce-style job:

    1. ``friend_pairs``: ``((user, friend), 1)`` for every direct friendship;
    2. emit ``((a, b), 1)`` for every ordered pair in each user's friend list
       (``a`` and ``b`` share that user as a mutual friend);
    3. drop pairs that are already friends, sum the rest -> mutual-friend counts;
    4. group by ``a`` and keep the ``top_k`` best ``b`` (count desc, id asc).
    """
    sc.addPyFile(str(Path(__file__).resolve()))  # workers import rank_candidates / parse_line
    wanted = None if users is None else set(users)
    wanted_bc = None if wanted is None else sc.broadcast(wanted)

    adjacency = (
        sc.textFile(str(path)).map(parse_line).filter(lambda parsed: parsed is not None).cache()
    )
    friend_pairs = adjacency.flatMap(lambda uf: [((uf[0], f), 1) for f in set(uf[1])])
    pairs = adjacency.flatMap(lambda uf: [(p, 1) for p in permutations(dict.fromkeys(uf[1]), 2)])
    if wanted_bc is not None:  # push the target filter in front of the expensive shuffle
        pairs = pairs.filter(lambda kv: kv[0][0] in wanted_bc.value)
    counts = pairs.subtractByKey(friend_pairs).reduceByKey(add)
    ranked = (
        counts.map(lambda kv: (kv[0][0], (kv[0][1], kv[1])))
        .groupByKey()
        .mapValues(lambda cands: rank_candidates(cands, top_k))
    )
    return dict(ranked.collect())


def write_recommendations(path: str | Path, recs: Recommendations) -> None:
    """``user<TAB>rec1,rec2,...`` sorted by user id."""
    with open(path, "w", encoding="utf8") as fh:
        for user in sorted(recs):
            fh.write(f"{user}\t{','.join(map(str, recs[user]))}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default=str(HERE / "sample_friends.txt"), help="adjacency list")
    parser.add_argument("--output", help="write 'user<TAB>recs' lines here (default: print only)")
    parser.add_argument("--users", type=int, nargs="*", help="only these users (default: all)")
    parser.add_argument("--top", type=int, default=10, help="recommendations per user")
    parser.add_argument("--engine", choices=["spark", "python"], default="spark")
    parser.add_argument("--master", default="local[2]", help="Spark master URL")
    parser.add_argument("--show", type=int, default=10, help="how many users to print")
    args = parser.parse_args(argv)

    if args.engine == "spark":
        recs = recommend_spark(get_spark_context(args.master), args.input, args.top, args.users)
    else:
        recs = recommend_reference(read_adjacency(args.input), args.top, args.users)

    print(f"{len(recs)} users have at least one recommendation (engine: {args.engine})")
    for user in sorted(recs)[: args.show]:
        print(f"{user}\t{','.join(map(str, recs[user]))}")
    if args.output:
        write_recommendations(args.output, recs)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
