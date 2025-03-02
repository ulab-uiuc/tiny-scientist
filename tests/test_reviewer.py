import pytest
from unittest.mock import Mock, patch
from tiny_scientist.reviewer import write_review, load_paper, get_review_fewshot_examples

@pytest.fixture
def mock_client():
    return Mock()

@pytest.fixture
def mock_model():
    return "gpt-4-test"

@pytest.fixture
def mock_prompts():
    return {
        "reviewer_system_prompt_neg": "Negative review prompt",
        "reviewer_system_prompt_pos": "Positive review prompt",
        "reviewer_system_prompt_base": "Base review prompt",
        "neurips_form": "NeurIPS form"
    }

@pytest.fixture
def mock_text():
    return "This is a test paper content."

@pytest.fixture
def mock_fewshot_papers():
    return ["/path/to/fewshot_paper1.pdf", "/path/to/fewshot_paper2.pdf"]

@pytest.fixture
def mock_fewshot_reviews():
    return ["/path/to/fewshot_review1.json", "/path/to/fewshot_review2.json"]


def test_write_review(mock_client, mock_model, mock_prompts, mock_text, mock_fewshot_papers, mock_fewshot_reviews):
    with patch('tiny_scientist.reviewer.get_review_fewshot_examples', return_value="Mock fewshot prompt"), \
         patch('tiny_scientist.reviewer.get_response_from_llm', return_value=("Mock LLM response", [])) as mock_llm, \
         patch('tiny_scientist.reviewer.extract_json_between_markers', return_value={"Summary": "Test summary", "Decision": "Accept"}):
        review = write_review(
            model=mock_model,
            client=mock_client,
            text=mock_text,
            reviewer_system_prompt=mock_prompts["reviewer_system_prompt_neg"],
            neurips_form=mock_prompts["neurips_form"],
            num_reflections=1,
            num_fs_examples=1,
            num_reviews_ensemble=1,
            return_msg_history=False,
            temperature=0.75
        )
        mock_llm.assert_called_once()
    assert review is not None


def test_load_paper():
    # This test would require a real PDF file or a mock of the file reading process
    pass


def test_get_review_fewshot_examples(mock_fewshot_papers, mock_fewshot_reviews):
    with patch('tiny_scientist.reviewer.load_paper', return_value="Mock paper text"), \
         patch('tiny_scientist.reviewer.load_review', return_value="Mock review text"):
        fewshot_prompt = get_review_fewshot_examples(
            num_fs_examples=1,
            fewshot_papers=mock_fewshot_papers,
            fewshot_reviews=mock_fewshot_reviews
        )
    assert "Below are some sample reviews" in fewshot_prompt
