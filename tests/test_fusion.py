from legalqa.fusion import reciprocal_rank_fusion


def test_bundle_members_share_dense_rank_in_rrf():
    fused = reciprocal_rank_fusion(
        sparse_ids=[10, 20],
        dense_ranked_ids=[(30, 1), (40, 1), (50, 2)],
        rrf_k=60,
    )
    scores = dict(fused)

    assert scores[30] == scores[40]
    assert scores[30] > scores[50]


def test_node_found_by_both_retrievers_is_boosted():
    fused = reciprocal_rank_fusion(
        sparse_ids=[10, 20],
        dense_ranked_ids=[(20, 1), (30, 2)],
        rrf_k=60,
    )

    assert fused[0][0] == 20
