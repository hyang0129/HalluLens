"""Fidelity tests: our SE pipeline produces the same outputs as the reference.

Reference: jlko/semantic_uncertainty
    semantic_uncertainty/uncertainty/uncertainty_measures/semantic_entropy.py

The reference's NLI model (microsoft/deberta-v2-xlarge-mnli) is a large
download; we don't run it in tests. Instead we synthesize NLI matrices and
log-probabilities, then check that our clustering, aggregation, and entropy
functions match the reference's algorithms bit-for-bit.

The vendored reference functions below are copied verbatim from the reference
repo (BSD-3 license, file noted above) and are the ground-truth against which
our implementations are compared.
"""
import numpy as np
import pytest

from tasks.sampling_baselines.se import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    cluster_assignment_entropy as ours_cluster_entropy,
    compute_se_for_record,
    get_semantic_ids as ours_get_semantic_ids,
    logsumexp_by_id as ours_logsumexp_by_id,
    predictive_entropy_rao as ours_predictive_entropy_rao,
)


# ---------------------------------------------------------------------------
# Vendored reference functions (jlko/semantic_uncertainty, BSD-3)
# Adapted minimally: `check_implication` is replaced by a closure over a
# precomputed argmax-implications matrix so we don't need to run the NLI model.
# ---------------------------------------------------------------------------

def ref_get_semantic_ids(strings_list, impl_matrix, strict_entailment=False):
    """Reference get_semantic_ids; impl_matrix[i,j] = argmax label for (i,j)."""
    def are_equivalent(i, j):
        implication_1 = int(impl_matrix[i, j])
        implication_2 = int(impl_matrix[j, i])
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        if strict_entailment:
            return (implication_1 == 2) and (implication_2 == 2)
        implications = [implication_1, implication_2]
        return (0 not in implications) and ([1, 1] != implications)

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for i, _ in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(i, j):
                    semantic_set_ids[j] = next_id
            next_id += 1
    assert -1 not in semantic_set_ids
    return semantic_set_ids


def ref_logsumexp_by_id(semantic_ids, log_likelihoods):
    """Reference logsumexp_by_id with agg='sum_normalized'."""
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    out = []
    for uid in unique_ids:
        idx = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_lp = [log_likelihoods[i] for i in idx]
        log_lik_norm = np.asarray(id_lp) - np.log(np.sum(np.exp(np.asarray(log_likelihoods))))
        out.append(np.log(np.sum(np.exp(log_lik_norm))))
    return out


def ref_predictive_entropy_rao(log_probs):
    lp = np.asarray(log_probs)
    return float(-np.sum(np.exp(lp) * lp))


def ref_cluster_assignment_entropy(semantic_ids):
    n = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probs = counts / n
    return float(-(probs * np.log(probs)).sum())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_nli_matrix_from_argmax(impl: np.ndarray) -> np.ndarray:
    """Build a (N, N, 3) prob matrix whose argmax is `impl` (deterministic)."""
    N = impl.shape[0]
    m = np.full((N, N, 3), 0.05, dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                m[i, j] = np.nan
                continue
            m[i, j, int(impl[i, j])] = 0.9
            # Renormalize to sum to 1 across the 3 classes
            m[i, j] /= m[i, j].sum()
    return m


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_two_clusters():
    """5 samples: indices {0,1,2} mutually entail; {3,4} mutually entail; cross = contradiction."""
    N = 5
    impl = np.full((N, N), NEUTRAL, dtype=int)
    cluster_a = [0, 1, 2]
    cluster_b = [3, 4]
    for i in cluster_a:
        for j in cluster_a:
            if i != j:
                impl[i, j] = ENTAILMENT
    for i in cluster_b:
        for j in cluster_b:
            if i != j:
                impl[i, j] = ENTAILMENT
    for i in cluster_a:
        for j in cluster_b:
            impl[i, j] = CONTRADICTION
            impl[j, i] = CONTRADICTION
    return impl


@pytest.fixture
def fixture_mixed_loose_vs_strict():
    """Pair that's `(entail, neutral)` — equivalent under loose, not under strict."""
    N = 3
    impl = np.full((N, N), NEUTRAL, dtype=int)
    impl[0, 1] = ENTAILMENT
    impl[1, 0] = NEUTRAL  # one-way only
    impl[0, 2] = CONTRADICTION
    impl[2, 0] = CONTRADICTION
    impl[1, 2] = NEUTRAL
    impl[2, 1] = NEUTRAL
    return impl


# --- clustering ---

def test_clustering_matches_reference_loose(fixture_two_clusters):
    impl = fixture_two_clusters
    nli = make_nli_matrix_from_argmax(impl)

    ours = ours_get_semantic_ids(nli, start_idx=0, end_idx=impl.shape[0], strict_entailment=False)
    ref = ref_get_semantic_ids(list(range(impl.shape[0])), impl, strict_entailment=False)
    assert ours == ref, f"loose clustering diverged: ours={ours} ref={ref}"
    # Sanity: two clusters
    assert len(set(ours)) == 2


def test_clustering_matches_reference_strict(fixture_two_clusters):
    impl = fixture_two_clusters
    nli = make_nli_matrix_from_argmax(impl)

    ours = ours_get_semantic_ids(nli, start_idx=0, end_idx=impl.shape[0], strict_entailment=True)
    ref = ref_get_semantic_ids(list(range(impl.shape[0])), impl, strict_entailment=True)
    assert ours == ref
    assert len(set(ours)) == 2


def test_strict_vs_loose_diverge_when_asymmetric(fixture_mixed_loose_vs_strict):
    impl = fixture_mixed_loose_vs_strict
    nli = make_nli_matrix_from_argmax(impl)
    N = impl.shape[0]

    loose_ours = ours_get_semantic_ids(nli, 0, N, strict_entailment=False)
    strict_ours = ours_get_semantic_ids(nli, 0, N, strict_entailment=True)
    loose_ref = ref_get_semantic_ids(list(range(N)), impl, strict_entailment=False)
    strict_ref = ref_get_semantic_ids(list(range(N)), impl, strict_entailment=True)

    assert loose_ours == loose_ref
    assert strict_ours == strict_ref
    assert loose_ours != strict_ours, "(entail, neutral) pair must split under strict"


def test_clustering_start_idx_skips_greedy(fixture_two_clusters):
    """SE skips index 0 (greedy answer). Ensure start_idx=1 produces ids over rows 1..N."""
    impl = fixture_two_clusters
    nli = make_nli_matrix_from_argmax(impl)
    N = impl.shape[0]

    ours_skip = ours_get_semantic_ids(nli, start_idx=1, end_idx=N, strict_entailment=False)
    # Equivalent to running the reference over only rows 1..N-1
    sub_impl = impl[1:, 1:]
    ref_skip = ref_get_semantic_ids(list(range(N - 1)), sub_impl, strict_entailment=False)
    assert ours_skip == ref_skip


# --- aggregation ---

def test_logsumexp_by_id_matches_reference():
    rng = np.random.default_rng(0)
    semantic_ids = [0, 0, 1, 2, 1, 2, 2]
    log_probs = rng.uniform(-5.0, -0.1, size=len(semantic_ids)).tolist()

    ours = ours_logsumexp_by_id(semantic_ids, log_probs)
    ref = ref_logsumexp_by_id(semantic_ids, log_probs)

    np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-10)


# --- entropy estimators ---

def test_predictive_entropy_rao_matches_reference():
    rng = np.random.default_rng(1)
    log_probs = (-np.abs(rng.standard_normal(5))).tolist()
    np.testing.assert_allclose(
        ours_predictive_entropy_rao(log_probs),
        ref_predictive_entropy_rao(log_probs),
        rtol=1e-12, atol=1e-12,
    )


def test_cluster_assignment_entropy_matches_reference():
    ids = [0, 0, 1, 2, 2, 2]
    np.testing.assert_allclose(
        ours_cluster_entropy(ids),
        ref_cluster_assignment_entropy(ids),
        rtol=1e-12, atol=1e-12,
    )


# --- end-to-end per-record ---

def test_compute_se_for_record_matches_reference_pipeline(fixture_two_clusters):
    """End-to-end: cluster + logsumexp + rao matches the reference pipeline."""
    impl = fixture_two_clusters
    # 5 stochastic samples + a greedy at index 0. impl is 5x5; embed into 6x6 with greedy.
    K = impl.shape[0]
    N = K + 1
    full_impl = np.full((N, N), NEUTRAL, dtype=int)
    full_impl[1:, 1:] = impl
    nli = make_nli_matrix_from_argmax(full_impl)

    rng = np.random.default_rng(2)
    seq_lp = (-np.abs(rng.standard_normal(K))).tolist()
    len_lp = (-np.abs(rng.standard_normal(K))).tolist()

    record = {
        "row_idx": 7,
        "K": K,
        "samples": [
            {"text": f"s{i}", "sequence_logprob": seq_lp[i], "length_normalized_logprob": len_lp[i]}
            for i in range(K)
        ],
    }
    out = compute_se_for_record(record, nli, strict_entailment=False)

    # Reference pipeline applied to the same K samples (skipping the greedy):
    ref_ids = ref_get_semantic_ids(list(range(K)), impl, strict_entailment=False)
    ref_cluster_lp = ref_logsumexp_by_id(ref_ids, seq_lp)
    expected_se = ref_predictive_entropy_rao(ref_cluster_lp)

    np.testing.assert_allclose(out["semantic_entropy"], expected_se, rtol=1e-10, atol=1e-12)
    assert out["n_clusters"] == len(set(ref_ids))
    assert out["semantic_ids"] == ref_ids
