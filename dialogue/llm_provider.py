import ollama


class LLMProvider:
    def __init__(self, prompt):
        self.prompt = prompt

    def query(self, prompt):
        raise NotImplementedError


class OpenAI(LLMProvider):
    def query(self, prompt):
        return "OpenAI"


class Replicate(LLMProvider):
    def query(self, prompt):
        return "Replicate"


class Ollama(LLMProvider):
    def query(self, prompt, model="llama2"):
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response["message"]["content"]


class HuggingFace(LLMProvider):
    def query(self, prompt):
        return "HuggingFace"


def llm_provider(prompt):
    return Ollama(prompt)
    # return HuggingFace(prompt)
    # return OpenAI(prompt)
    # return Replicate(prompt)
    # return Ollama(prompt)
    # return HuggingFace(prompt)
    # return OpenAI(prompt)
    # return Replicate(prompt)
    # return Ollama(prompt)
    # return HuggingFace(prompt)
    # return OpenAI(prompt)
    # return Replicate(prompt)
    # return Ollama(prompt)
    # return HuggingFace(prompt)
    # return OpenAI(prompt)
    # return Replicate(prompt)
    # return Ollama(prompt)
    # return HuggingFace(prompt)
    # return OpenAI(prompt)
    # return Replicate(prompt)
    # return Ollama(prompt)
    # return HuggingFace(prompt)
    # return OpenAI(prompt)
    # return Replicate(prompt)
    # return Ollama(prompt)
