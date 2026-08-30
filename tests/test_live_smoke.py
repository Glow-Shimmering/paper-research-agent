from types import SimpleNamespace

from scripts.smoke_live_deepseek import _ensure_review_question_id


class _QuestionRepository:
    def __init__(self, questions=()):
        self.questions = tuple(questions)
        self.created = 0

    def list_questions(self, project_id):
        assert project_id == "project-1"
        return self.questions

    def create_question(self, project_id, question):
        assert project_id == "project-1"
        assert question == "这些论文的共同点是什么？"
        self.created += 1
        item = SimpleNamespace(id="question-created")
        self.questions = (item,)
        return item


def test_live_smoke_reuses_first_tuple_question():
    repository = _QuestionRepository([SimpleNamespace(id="question-existing")])

    assert _ensure_review_question_id(repository, "project-1") == "question-existing"
    assert repository.created == 0


def test_live_smoke_creates_question_for_empty_tuple():
    repository = _QuestionRepository()

    assert _ensure_review_question_id(repository, "project-1") == "question-created"
    assert repository.created == 1
