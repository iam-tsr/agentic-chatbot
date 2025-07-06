from langchain_core.tools import tool

@tool
def addition(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a + b