from langchain_core.tools import tool


@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b


@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b


@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b


# Keep the graph tool surface unchanged for now. The demo tools above are
# retained for quick experiments, but are not exposed to the agent.
tools = []
