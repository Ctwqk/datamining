import random
import shutil

import pytest

from friend_recommendation_spark import (
    HERE,
    main,
    parse_line,
    rank_candidates,
    read_adjacency,
    recommend_reference,
    write_recommendations,
)
from make_sample_graph import make_graph, write_graph

SAMPLE = HERE / "sample_friends.txt"

# 1-2, 1-3, 2-3, 2-4; user 5 has no friends
TOY = {1: [2, 3], 2: [1, 3, 4], 3: [1, 2], 4: [2], 5: []}


def brute_force(graph: dict[int, list[int]], user: int, top_k: int) -> list[int]:
    """Independent definition for symmetric graphs: friends-of-friends, not friends, not self."""
    mutual: dict[int, int] = {}
    for friend in graph[user]:
        for fof in graph[friend]:
            if fof != user and fof not in graph[user]:
                mutual[fof] = mutual.get(fof, 0) + 1
    order = sorted(mutual.items(), key=lambda kv: (-kv[1], kv[0]))
    return [v for v, _ in order[:top_k]]


def random_symmetric_graph(n=60, p=0.08, seed=0):
    rng = random.Random(seed)
    friends = {u: set() for u in range(n)}
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                friends[u].add(v)
                friends[v].add(u)
    return {u: sorted(f) for u, f in friends.items()}


def test_parse_line():
    assert parse_line("3\t1,2,5\n") == (3, [1, 2, 5])
    assert parse_line("5\t\n") == (5, [])  # user with no friends
    assert parse_line("7\n") is None  # no tab -> skipped, as in the original job
    assert parse_line("\n") is None


def test_rank_candidates_orders_by_count_then_id():
    assert rank_candidates([(9, 1), (4, 3), (2, 3), (7, 2)], top_k=3) == [2, 4, 7]


def test_reference_on_toy_graph_excludes_existing_friends():
    recs = recommend_reference(TOY)
    assert recs[1] == [4]  # 4 is a friend of 2; 2 and 3 are already friends of 1
    assert recs[3] == [4]
    assert recs[4] == [1, 3]  # both share friend 2 with user 4
    assert 5 not in recs  # nobody to recommend


def test_reference_never_recommends_self_or_friends():
    graph = random_symmetric_graph()
    for user, recs in recommend_reference(graph).items():
        assert user not in recs
        assert not set(recs) & set(graph[user])
        assert len(recs) == len(set(recs))


def test_reference_matches_brute_force_and_top_k():
    graph = random_symmetric_graph(seed=4)
    for top_k in (1, 3, 10):
        recs = recommend_reference(graph, top_k=top_k)
        for user in graph:
            expected = brute_force(graph, user, top_k)
            assert recs.get(user, []) == expected


def test_reference_user_filter():
    recs = recommend_reference(TOY, users=[4])
    assert list(recs) == [4]


def test_sample_graph_is_symmetric_and_reproducible(tmp_path):
    graph = read_adjacency(SAMPLE)
    assert len(graph) == 1000
    assert all(u in graph[v] for u, fs in graph.items() for v in fs)
    assert make_graph(seed=0) == graph  # the committed file is what the script generates
    write_graph(tmp_path / "again.txt", make_graph(seed=0))
    assert (tmp_path / "again.txt").read_text() == SAMPLE.read_text()


def test_cli_python_engine_writes_output(tmp_path, capsys):
    out = tmp_path / "recs.tsv"
    assert main(["--engine", "python", "--users", "11", "924", "--output", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert [ln.split("\t")[0] for ln in lines] == ["11", "924"]
    assert "engine: python" in capsys.readouterr().out
    graph = read_adjacency(SAMPLE)
    assert lines[0].split("\t")[1] == ",".join(map(str, brute_force(graph, 11, 10)))


def test_write_recommendations_format(tmp_path):
    path = tmp_path / "r.tsv"
    write_recommendations(path, {2: [9, 8], 1: [3]})
    assert path.read_text() == "1\t3\n2\t9,8\n"


# ----------------------------------------------------------------------------- PySpark


@pytest.fixture(scope="module")
def spark_context():
    pytest.importorskip("pyspark")
    if shutil.which("java") is None:
        pytest.skip("no JDK on PATH")
    from friend_recommendation_spark import get_spark_context

    sc = get_spark_context("local[2]")
    yield sc
    sc.stop()


@pytest.mark.spark
def test_spark_matches_reference_on_toy_graph(spark_context, tmp_path):
    from friend_recommendation_spark import recommend_spark

    path = tmp_path / "toy.txt"
    write_graph(path, TOY)
    assert recommend_spark(spark_context, path) == recommend_reference(TOY)


@pytest.mark.spark
def test_spark_does_not_recommend_existing_friends(spark_context, tmp_path):
    """Regression test for the original job, which returned user 1 -> [2, 3, 4]."""
    from friend_recommendation_spark import recommend_spark

    path = tmp_path / "toy.txt"
    write_graph(path, TOY)
    recs = recommend_spark(spark_context, path)
    assert recs[1] == [4]


@pytest.mark.spark
def test_spark_matches_reference_on_sample_graph(spark_context):
    from friend_recommendation_spark import recommend_spark

    expected = recommend_reference(read_adjacency(SAMPLE))
    assert recommend_spark(spark_context, SAMPLE) == expected
    only = recommend_spark(spark_context, SAMPLE, users=[11, 924], top_k=5)
    assert only == {u: expected[u][:5] for u in (11, 924)}
