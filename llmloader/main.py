import typer
from llmloader import load
from rich.console import Console


app = typer.Typer()

@app.command()
def main(
    prompt:str = typer.Argument(help="Prompt for the model"),
    model:str = typer.Option("gpt-4o-mini", help="Model Name"),
    temperature:float = typer.Option(0.1, help="Temperature for sampling"),
    max_tokens:int=typer.Option(None, help="Max number of tokens to generate"),
    api_key:str=typer.Option("", help="API Key for the model"),
    all_results:bool=typer.Option(False, help="Print all results"),
):
    llm = load(model=model, temperature=temperature, api_key=api_key, max_tokens=max_tokens)
    result = llm.invoke(prompt)
    
    console = Console()
    console.print(result.content)
    
    if all_results:
        console.print(result)