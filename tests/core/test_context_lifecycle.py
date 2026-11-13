import gc
import weakref

import minisgl.core as core
import pytest


def test_context_release_allows_collection_and_next_engine(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    ctx = core.Context(page_size=1)
    reference = weakref.ref(ctx)
    core.set_global_ctx(ctx)
    core.clear_global_ctx(ctx)
    del ctx
    gc.collect()
    assert reference() is None
    with pytest.raises(AssertionError, match="not set"):
        core.get_global_ctx()
    replacement = core.Context(page_size=1)
    core.set_global_ctx(replacement)
    assert core.get_global_ctx() is replacement
    core.clear_global_ctx(replacement)


def test_context_release_rejects_active_forward_or_wrong_owner(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    ctx = core.Context(page_size=1)
    core.set_global_ctx(ctx)
    with pytest.raises(AssertionError, match="another engine"):
        core.clear_global_ctx(core.Context(page_size=1))
    with ctx.forward_batch(core.Batch([], "decode")):
        with pytest.raises(AssertionError, match="during a forward"):
            core.clear_global_ctx(ctx)
        assert core.get_global_ctx() is ctx
    core.clear_global_ctx(ctx)
