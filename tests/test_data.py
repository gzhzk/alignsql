"""Tests for data processing modules."""

from alignsql.data.preprocessing import classify_difficulty, build_sft_prompt


def test_classify_difficulty_easy():
    assert classify_difficulty("SELECT name FROM users WHERE id = 1") == "easy"


def test_classify_difficulty_medium():
    assert classify_difficulty("SELECT COUNT(*) FROM users") == "medium"
    assert classify_difficulty("SELECT name FROM users GROUP BY age") == "medium"
    assert classify_difficulty("SELECT name FROM users ORDER BY id") == "medium"


def test_classify_difficulty_hard():
    assert classify_difficulty("SELECT * FROM a JOIN b ON a.id = b.id") == "hard"
    assert classify_difficulty("SELECT * FROM a WHERE a.x IN (SELECT y FROM b)") == "hard"


def test_classify_difficulty_extra():
    assert classify_difficulty("SELECT name FROM users UNION SELECT name FROM admins") == "extra"


def test_build_sft_prompt():
    schema = "Database: test\nTable t: id, name"
    question = "Find all names"
    prompt = build_sft_prompt(schema, question)
    assert "###Input:" in prompt
    assert schema in prompt
    assert question in prompt
    assert "###Response:" in prompt
