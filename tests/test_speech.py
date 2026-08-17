"""Sentence chunking for streamed replies."""

from __future__ import annotations

import pytest

from roxy.speech import FIRST_MIN_CHARS, MAX_CHARS, sentence_chunks


def as_deltas(text: str, size: int = 3):
    """Simulate token-by-token arrival."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def test_nothing_is_lost():
    text = "It's about twenty degrees outside. Warmer than yesterday. Bring a jacket anyway."
    joined = " ".join(sentence_chunks(as_deltas(text)))
    assert joined.split() == text.split()


def test_splits_on_sentence_ends():
    text = "The sky scatters blue light. That is why it looks blue. Simple enough."
    chunks = list(sentence_chunks(as_deltas(text)))
    assert len(chunks) >= 2
    assert chunks[0] == "The sky scatters blue light."


def test_first_chunk_comes_out_early():
    # A short opener should be emitted rather than held back waiting for more,
    # because time-to-first-audio is the whole point.
    chunks = list(sentence_chunks(as_deltas("Sure thing, happy to help. " + "x" * 300)))
    assert chunks[0] == "Sure thing, happy to help."


def test_decimals_are_not_sentence_ends():
    text = "It is 3.5 degrees and the humidity is 82.4 percent right now outside."
    # No full stop until the end, so no *strong* cut -- but the first chunk is
    # allowed a weak one, so assert on where it breaks, not on the count.
    assert not any(c.rstrip().endswith("3.5") for c in sentence_chunks(as_deltas(text)))


def test_single_sentence_reply_still_starts_speaking_early():
    # The case that made streaming pointless before: one long sentence with no
    # full stop until the very end. The first clause must come out on its own.
    text = ("A black hole is a region in space where gravity is so strong that "
            "nothing, not even light, can escape from it.")
    chunks = list(sentence_chunks(as_deltas(text)))
    assert len(chunks) > 1
    assert chunks[0].endswith(",")
    assert " ".join(chunks).split() == text.split()


def test_only_the_first_chunk_breaks_on_commas():
    # Later chunks wait for real sentence ends, so prosody holds together.
    text = ("First one ends here. Then a long second sentence, with commas in it, "
            "that should not be chopped at every one of them.")
    chunks = list(sentence_chunks(as_deltas(text)))
    assert chunks[0] == "First one ends here."
    assert not any(c.endswith(",") for c in chunks[1:])


def test_closing_quote_stays_with_its_sentence():
    text = 'She said "hello there." Then she left the room without saying more.'
    chunks = list(sentence_chunks(as_deltas(text)))
    assert chunks[0].endswith('"')


def test_runaway_sentence_is_broken_up():
    # No sentence end at all: must still split, or the first audio waits for
    # the entire reply.
    text = "well " * 120
    chunks = list(sentence_chunks(as_deltas(text)))
    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHARS + 40 for c in chunks)


def test_empty_input_yields_nothing():
    assert list(sentence_chunks([])) == []
    assert list(sentence_chunks(["", ""])) == []


def test_whitespace_only_yields_nothing():
    assert list(sentence_chunks(["   ", "\n"])) == []


def test_single_short_reply_is_emitted():
    assert list(sentence_chunks(as_deltas("Yes."))) == ["Yes."]


@pytest.mark.parametrize("size", [1, 2, 5, 17, 500])
def test_chunking_is_independent_of_delta_size(size):
    text = "First sentence here. Second sentence follows it. Third one ends things."
    chunks = list(sentence_chunks(as_deltas(text, size)))
    assert " ".join(chunks).split() == text.split()


def test_first_min_is_respected_for_tiny_openers():
    # "Hi." is shorter than FIRST_MIN_CHARS, so it should be merged with what
    # follows rather than sent to TTS on its own.
    assert FIRST_MIN_CHARS > 4
    chunks = list(sentence_chunks(as_deltas("Hi. The time is half past three now.")))
    assert chunks[0].startswith("Hi. The time")
