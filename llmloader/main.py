import typer
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console

from llmloader import LLMWrapper, load

app = typer.Typer()


@app.command()
def main(
    prompt: str = typer.Argument(help="Prompt for the model"),
    model: str = typer.Option("gpt-6-luna", help="Model Name"),
    temperature: float = typer.Option(None, help="Temperature for sampling (defaults to the model's own default)"),
    max_tokens: int = typer.Option(None, help="Max number of tokens to generate"),
    api_key: str = typer.Option("", help="API Key for the model"),
    endpoint: str = typer.Option("", help="Endpoint for the model"),
    all_results: bool = typer.Option(False, help="Print all results"),
    count: bool = typer.Option(False, help="Count tokens in the response"),
):
    llm = load(model=model, temperature=temperature, api_key=api_key, max_tokens=max_tokens, endpoint=endpoint)
    result = llm.invoke(prompt)

    console = Console()

    parser = StrOutputParser()
    console.print(parser.invoke(result))

    if count:
        console.print(LLMWrapper.get_token_count(result.response_metadata))

    if all_results:
        console.print(result)
