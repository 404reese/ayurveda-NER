import os
import sys

import pytest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache(tmp_path_factory):
    os.environ["AYURNER_CACHE_DIR"] = str(tmp_path_factory.mktemp("ayurner-cache"))
    os.environ.pop("AYURNER_LEXICONS", None)
    yield


@pytest.fixture(scope="session")
def ner():
    import ayurner

    return ayurner.load()


def spans(doc):
    return [(e.text, e.label) for e in doc.entities]
