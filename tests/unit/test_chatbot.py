from dialogue.chatbot import get_message


def test_get_message():
    query = "Hello, how are you?"
    response = get_message(query)
    assert isinstance(response, str), "The response should be a string"
