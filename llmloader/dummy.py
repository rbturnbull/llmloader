from collections.abc import Sequence
from langchain_core.language_models.llms import LLM
from .loader import Loader

from collections.abc import Sequence
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult


class DummyLLM(LLM):
    """A dummy LLM that just returns the input prompt in .content."""

    def _call(
        self,
        prompt: str,
        stop: str | Sequence[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        # For backwards compatibility: return raw string
        return prompt

    @property
    def _identifying_params(self) -> dict[str, str]:
        return {}

    @property
    def _llm_type(self) -> str:
        return "dummy"


class DummyLoader(Loader):
    def __call__(
        self,
        model: str,
        *args,
        **kwargs,
    ) -> LLM | None:
        if not model == 'dummy':
            return None

        return DummyLLM()
