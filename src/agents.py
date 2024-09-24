import os
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .utils import extract_sql_query, extract_python_code
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import pandas as pd
import altair as alt
from langchain_experimental.utilities import PythonREPL


class PlotGenerator:
    def __init__(self, model):
        load_dotenv()
        self.api_key = os.getenv("OPENWEBUI_API_KEY")
        self.llm = None
        self.setup_llm(model)
        self.python_repl = PythonREPL()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def setup_llm(self, model):
        key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=model, openai_api_key=key)

    def generate_plotting_code(
        self,
        question: str,
        df: pd.DataFrame,
        query: str,
        explanation: str,
        previous_code: str | None,
        previous_error: str | None,
    ) -> dict:
        prompt = f"""
        You are a data visualization expert. Given the following information:
        
        User question: {question}
        SQL query: {query}
        Explanation: {explanation}
        DataFrame: {df.to_string()}
        
        Your task is to create an appropriate Altair visualization for this data. Follow these steps:
        - Redefine the 'df' object explicitely. 
        - Analyze the data and the user's question to determine the most suitable chart type.
        - Write Python code using Altair to create the visualization.
        - Make sure to handle potential errors, such as column names with spaces.
        - Use appropriate colors, labels, and titles to make the chart informative and visually appealing.
        - Print the Altair chart object (name it 'chart') so it can be captured and displayed.

        Return only the Python code for creating the Altair chart.
        """

        # Convert DataFrame to dict for JSON serialization
        if previous_error is not None:
            response = self.llm.invoke(
                prompt
                + f"\n The code ```{previous_code}``` raised ```{previous_error}```. Please correct yourself. New code:\n"
            )
        else:
            response = self.llm.invoke(prompt)
        return extract_python_code(response.content)

    def generate_plot(self, question, df, query, explanation, max_attempts=3):
        attempt = 0
        previous_code = None
        previous_error = None
        chart = None
        while attempt <= max_attempts:
            attempt += 1
            plotting_code = self.generate_plotting_code(
                question=question,
                df=df,
                query=query,
                explanation=explanation,
                previous_code=previous_code,
                previous_error=previous_error,
            )
            try:
                exec_globals = {}
                exec(plotting_code, exec_globals)
                chart = exec_globals.get("chart", None)
                return {
                    "chart": chart,
                    "code": plotting_code,
                    "error": previous_error,
                    "attempts": attempt,
                }
            except Exception as e:
                attempt += 1
                previous_error = str(
                    e
                )  # potentiellement réduire l'erreur à son titre pour éviter de surcharger le prompt
                previous_code = plotting_code

        return {
            "chart": chart,
            "code": plotting_code,
            "error": previous_error,
            "attempts": attempt,
        }


class DBDescriptor:
    def __init__(self, db_url, model):
        load_dotenv()
        self.api_key = os.getenv("OPENWEBUI_API_KEY")
        self.llm = None
        self.engine = create_engine(url=db_url)
        self.db = SQLDatabase(engine=self.engine, schema="bdschema")
        self.setup_llm(model)
        self.descriptor_chain = None
        self.setup_descriptor_chain()

    def setup_llm(self, model):
        key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=model, openai_api_key=key)

    def setup_descriptor_chain(self):
        system = """
        You are a data analysis and {dialect} expert tasked with generating natural language descriptions of databases. 
        You will receive the table structure and schema details.

        Your job is to:
        - Write a very concise description of the database.
        - List the tables and their keys (with description).
        - Explain which keys are shared between tables if you have the information, otherwise mention that you don't know them.
        
        Informations:
        ---
        **Table Info**: 
        {table_info}

        **Your report**: 
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{dialect} {table_info}"),
            ]
        )

        self.descriptor_chain = prompt | self.llm | StrOutputParser()

    def describe_database(self):
        return self.descriptor_chain.stream(
            {"dialect": self.db.dialect, "table_info": self.db.table_info}
        )


class SQLAgent:
    def __init__(self, db_url, model):
        load_dotenv()
        self.api_key = os.getenv("OPENWEBUI_API_KEY")
        self.llm = None
        self.engine = create_engine(url=db_url)
        self.db = SQLDatabase(engine=self.engine, schema="bdschema")
        self.setup_llm(model)
        self.developer_chain = None
        self.setup_dev_chain()
        self.execute_query = QuerySQLDataBaseTool(db=self.db, verbose=True)
        self.reporter_chain = None
        self.setup_reporter_chain()

    def setup_llm(self, model):
        key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=model, openai_api_key=key)

    def setup_dev_chain(self):
        system = """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
            Unless the user specifies in the question a specific number of examples to obtain, query all the results. Otherwise, query {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
            Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
            Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            Pay attention to use date('now') function to get the current date, if the question involves "today".

            Only use the following tables:
            {table_info}

            Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
            - Using NOT IN with NULL values
            - Using UNION when UNION ALL should have been used
            - Using BETWEEN for exclusive ranges
            - Data type mismatch in predicates
            - Properly quoting identifiers
            - Using the correct number of arguments for functions
            - Casting to the correct data type
            - Using the proper columns for joins

            Use format:

            First draft: <<FIRST_DRAFT_QUERY>>
            Final answer: <<FINAL_ANSWER_QUERY>>
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "{input}")]
        ).partial(dialect=self.db.dialect)

        # def parse_final_answer(output: str) -> str:
        #     return output.split("Final answer: ")[1]

        self.developer_chain = (
            create_sql_query_chain(self.llm, self.db, prompt=prompt) | StrOutputParser()
        )

    def setup_reporter_chain(self):
        reporter_system = """
        You are a data analysis expert tasked with generating natural language descriptions of SQL query results. You will receive the following information:
        1. The user's question.
        2. The table structure and schema details.
        3. The SQL query used to generate the results.
        4. The actual results of the query.

        Your job is to:
        - Write a very concise report of the results in natural language.
        - Provide short clarification if the results are complex or not straightforward (complicated query).
        - Use simple and clear language so that even non-experts can understand the data.
        
        Informations:
        ---
        **Question**: {user_question}

        **SQL Query**:
        ```sql
        {sql_query}
        ```

        **Table Info**: 
        {table_info}

        **Query Results**: 
        {query_results}

        **Your report**: 
        """

        reporter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", reporter_system),
                ("human", "{user_question} {sql_query} {table_info} {query_results}"),
            ]
        )

        self.reporter_chain = reporter_prompt | self.llm | StrOutputParser()

    def process(self, question, max_attempts):
        attempt = 0
        queries = []
        ask_the_dev = question
        while attempt < max_attempts:
            try:
                query = extract_sql_query(
                    self.developer_chain.invoke({"question": ask_the_dev})
                )
                queries.append(query)
                results = pd.read_sql_query(query, con=self.engine)

                if not results.empty:
                    # c'est ok
                    answer = self.reporter_chain.invoke(
                        {
                            "user_question": question,
                            "sql_query": query,
                            "table_info": self.db.table_info,
                            "query_results": results.to_string(),
                        }
                    )
                    return {
                        "query": query,
                        "results": results,
                        "answer": answer,
                        "attempts": attempt,
                    }
                else:
                    # reboucler tant qu'il trouve pas de résultat. Reformuler question?
                    attempt += 1
                    if attempt <= max_attempts:
                        ask_the_dev = f"""Your previous queries 
                        ({", ".join(f"```{item}```" for item in queries)}) returned nothing after {attempt+1} attempts,
                        you may have misunterpreted the question. Try again: {question} """
                    else:
                        answer = self.reporter_chain.invoke(
                            {
                                "user_question": question,
                                "sql_query": query,
                                "table_info": self.db.table_info,
                                "query_results": f"Failed to find relevant results after {max_attempts} attempts.",
                            }
                        )
                        return {
                            "query": query,
                            "results": results,
                            "answer": answer,
                            "attempts": attempt,
                        }

            except Exception as e:
                attempt += 1
                error_message = str(e)
                if attempt <= max_attempts:
                    # rappeler la dev chain en mentionnant l'erreur
                    # OU reformuler la question ?
                    ask_the_dev = f"Your previous query (```{query}```) was incorrect. {question} "
                else:
                    answer = self.reporter_chain.invoke(
                        {
                            "user_question": question,
                            "sql_query": query,
                            "table_info": self.db.table_info,
                            "query_results": f"Failed to query the database after {max_attempts} attempts. Last error: {error_message}",
                        }
                    )
                    return {
                        "query": query,
                        "results": None,
                        "answer": answer,
                        "attempts": attempt,
                    }

                question = f"The previous query resulted in an error: '{error_message}'. The query was: '{query}'. Please correct the query and try again."

        answer = self.reporter_chain.invoke(
            {
                "user_question": question,
                "sql_query": None,
                "table_info": self.db.table_info,
                "query_results": f"Failed to query the database after {max_attempts} attempts. Provide an excuse message and try to justify. ",
            }
        )
        return {
            "query": query,
            "results": None,
            "answer": answer,
            "attempts": attempt,
        }
