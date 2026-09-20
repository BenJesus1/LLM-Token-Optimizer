"""Smoke tests that the installable packages import under Python 3.11+."""


def test_allocator_imports() -> None:
    import allocator

    assert allocator.__doc__


def test_benchmarks_imports() -> None:
    import benchmarks

    assert benchmarks.__doc__
