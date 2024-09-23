import re


def add_schema_to_query(schema: str, sql_query: str) -> str:
    table_pattern = r"(FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z_][a-zA-Z0-9_]*)"

    def replace_with_schema(match):
        return f"{match.group(1)} {schema}.{match.group(2)}"

    modified_query = re.sub(
        table_pattern, replace_with_schema, sql_query, flags=re.IGNORECASE
    )

    return modified_query


def extract_sql_query(llm_output):
    """
    Extracts the SQL query from the LLM output using regex.

    Args:
        llm_output (str): The output from the LLM containing a SQL query.

    Returns:
        str: The extracted SQL query, or None if no query is found.
    """
    # Regex pattern to capture text inside ```sql ... ``` blocks
    sql_pattern = r"```sql\s*(.*?)\s*```"

    # Search for the pattern in the LLM output
    match = re.search(sql_pattern, llm_output, re.DOTALL)

    if match:
        # If match is found, return the SQL query
        return match.group(1).strip()

    # Fallback if no ```sql``` blocks are found
    # This pattern looks for SELECT/INSERT/UPDATE/DELETE queries in the text
    sql_fallback_pattern = r"(SELECT|INSERT|UPDATE|DELETE).*?;"
    fallback_match = re.search(
        sql_fallback_pattern, llm_output, re.IGNORECASE | re.DOTALL
    )

    if fallback_match:
        return fallback_match.group(0).strip()

    # If no query found, return None
    return None


def extract_python_code(llm_output):
    """
    Extracts Python code from the LLM output using regex.

    Args:
        llm_output (str): The output from the LLM containing Python code.

    Returns:
        str: The extracted Python code, or None if no code is found.
    """
    # Regex pattern to capture text inside ```python ... ``` blocks
    python_pattern = r"```python\s*(.*?)\s*```"

    # Search for the pattern in the LLM output
    match = re.search(python_pattern, llm_output, re.DOTALL)

    if match:
        # If match is found, return the Python code
        return match.group(1).strip()

    # Fallback if no ```python``` blocks are found
    # This pattern looks for function definitions and common Python syntax
    python_fallback_pattern = r"(def|class|import).*"
    fallback_match = re.search(
        python_fallback_pattern, llm_output, re.IGNORECASE | re.DOTALL
    )

    if fallback_match:
        return fallback_match.group(0).strip()

    # If no code found, return None
    return None
