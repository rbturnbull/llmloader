import typer
from llmloader import load

app = typer.Typer()

@app.command()
def main(
    prompt:str = typer.Argument(help="Prompt for the model"),
    model:str = typer.Option("gpt-4o", help="Model Name"),
    temperature:float = typer.Option(None, help="Temperature for sampling"),
    max_tokens:int=typer.Option(None, help="Max number of tokens to generate"),
    api_key:str=typer.Option("", help="API Key for the model"),
):
    llm = load(model, temperature=temperature, api_key=api_key, max_tokens=max_tokens)
    print(llm(prompt))