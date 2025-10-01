import os

from app.document_processor_hybrid import process_image_path


def test_process_nonexistent_file():
    try:
        process_image_path("tests/does_not_exist.jpg")
    except FileNotFoundError:
        assert True
    else:
        assert False, "Should raise FileNotFoundError for missing file"

