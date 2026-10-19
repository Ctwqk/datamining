"""Generate ``sample_friends.txt``: a small synthetic friendship graph.

The real assignment data (a LiveJournal-style friend list) is not part of this
repository, so this script builds a reproducible stand-in: ten communities of 100 users
with dense friendships inside a community and a few friendships across communities
(a stochastic block model). The output uses the assignment's format,
``user<TAB>friend1,friend2,...``, with symmetric friendships.

    python make_sample_graph.py            # rewrites sample_friends.txt (seed 0)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def make_graph(
    n_users: int = 1000,
    n_communities: int = 10,
    p_in: float = 0.12,
    p_out: float = 0.002,
    seed: int = 0,
) -> dict[int, list[int]]:
    rng = random.Random(seed)
    community = [u * n_communities // n_users for u in range(n_users)]
    friends: dict[int, set[int]] = {u: set() for u in range(n_users)}
    for u in range(n_users):
        for v in range(u + 1, n_users):
            if rng.random() < (p_in if community[u] == community[v] else p_out):
                friends[u].add(v)
                friends[v].add(u)
    return {u: sorted(fs) for u, fs in friends.items()}


def write_graph(path: str | Path, graph: dict[int, list[int]]) -> None:
    with open(path, "w", encoding="utf8") as fh:
        for user in sorted(graph):
            fh.write(f"{user}\t{','.join(map(str, graph[user]))}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(HERE / "sample_friends.txt"))
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    graph = make_graph(args.users, seed=args.seed)
    write_graph(args.out, graph)
    edges = sum(len(v) for v in graph.values()) // 2
    print(f"wrote {args.out}: {len(graph)} users, {edges} friendships")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
