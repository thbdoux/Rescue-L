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
from sqlalchemy import create_engine, text

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import pandas as pd
import altair as alt
from langchain_experimental.utilities import PythonREPL
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List


class PlotGenerator:
    def __init__(self, model):
        load_dotenv()
        self.api_key = os.getenv("OPENWEBUI_API_KEY")
        self.llm = None
        self.setup_llm(model)
        self.python_repl = PythonREPL()

    def setup_llm(self, model):
        key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=model, openai_api_key=key, temperature=0)

    def generate_plotting_code(
        self,
        question: str,
        df: pd.DataFrame,
        query: str,
        explanation: str,
        previous_code: str | None,
        previous_error: str | None,
    ) -> dict:

        df_info = (
            (df if len(df) < 10 else df.sample(10)).to_string()
            if type(df) == pd.DataFrame
            else df
        )
        prompt = f"""
        You are a data visualization expert. Given the following information:
        
        User question: {question}
        SQL query: {query}
        Explanation: {explanation}
        DataFrame (called 'df' in the code, already defined): {df_info}
        
        Your task is to create an appropriate Altair visualization for this data. Follow these steps:
        - 
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
                exec_globals = {"df": df}
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


class PandasAgent:
    def __init__(self, model, df=None):
        load_dotenv()
        self.llm = None
        self.setup_llm(model)
        self.df = df["df"]
        self.df_schema = self.df.info  # df["schema"]
        self.question_generator = None
        self.setup_question_generator()
        self.pandas_dev_chain = None
        self.setup_pandas_dev_chain()
        self.reporter_chain = None
        self.setup_reporter_chain()

    def setup_llm(self, model):
        key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=model, openai_api_key=key, temperature=0)

    def setup_question_generator(self):
        class Kpis(BaseModel):
            kpis: List[str] = Field(description="List of KPIs")

        parser = JsonOutputParser(pydantic_object=Kpis)
        self.question_generator_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert (sales, marketing...) who wants to explore a company database.
            Given a database schema, generate a list of relevant KPI for this database.
            Always begin by main KPIs like : global revenue, profit, margin, sales distribution.
            And generate more if you think it is relevant.
            You must be sure to find relevant data according to the db schema. 
            
            Format your response as : {format_instructions}""",
                ),
                (
                    "user",
                    "Schema:\n{schema}\n\nWrite between 5 to 8 queries. .",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        self.question_generator = self.question_generator_prompt | self.llm | parser

    def setup_pandas_dev_chain(self):
        class PandasQueryOutput(BaseModel):
            """Output schema for the pandas query generation."""

            reasoning: str = Field(
                description="Step-by-step reasoning about how to solve the query"
            )
            pandas_query: str = Field(description="The actual pandas query to execute")
            safety_checks: List[str] = Field(
                description="List of safety checks to perform before executing the query"
            )

        # Create the parser
        parser = JsonOutputParser(pydantic_object=PandasQueryOutput)

        # Define the template
        system = """You are an expert data analyst specialized in generating pandas queries. Your task is to generate safe and efficient pandas code based on the user's request.
        DataFrame three first lines:
        {df_schema}

        Instructions:
        - Assume 'df' is already defined.
        - Analyze the info carefully to understand available columns and their types
        - Think step by step about how to solve the query
        - Generate appropriate python code with pandas.
        - Include necessary safety checks (null values, type checking, etc.)
        - Optimize for readability and performance
        - ALWAYS store the result of the query in a variable called 'res' so it can be captured. 'res' is always a pandas dataframe. ALWAYS !

        Constraints:
        - Always use proper column names as shown in the schema
        - Include error handling where appropriate
        - Avoid operations that might cause memory issues
        - Prefer vectorized operations over loops
        - Use proper data types for operations

        {format_instructions}
        """

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "{input}")]
        ).partial(
            format_instructions=parser.get_format_instructions(),
            df_schema=self.df.info,
        )
        self.pandas_dev_chain = prompt | self.llm | parser

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
            attempt += 1
            try:
                out = self.pandas_dev_chain.invoke(ask_the_dev)
                reasoning, pandas_query, safety_checks = (
                    out["reasoning"],
                    out["pandas_query"],
                    out["safety_checks"],
                )

                queries.append(pandas_query)
                # exec python code
                exec_globals = {"df": self.df}
                exec(pandas_query, exec_globals)
                res = exec_globals.get("res", None)
                res_preview = (
                    (res.sample(20).to_string() if len(res) >= 20 else res.to_string())
                    if type(res) == pd.DataFrame
                    else res
                )
                answer = self.reporter_chain.invoke(
                    {
                        "user_question": question,
                        "sql_query": pandas_query,
                        "table_info": self.df.info,
                        "query_results": res_preview,
                    }
                )
                return {
                    "query": pandas_query,
                    "results": res,
                    "answer": answer,
                    "attempts": attempt,
                }
            except Exception as e:
                attempt += 1
                error_message = str(e)
                if attempt <= max_attempts:
                    # rappeler la dev chain en mentionnant l'erreur
                    # OU reformuler la question ?
                    ask_the_dev = f"Your previous query (```{pandas_query}```) was incorrect. {question} "
                else:
                    answer = self.reporter_chain.invoke(
                        {
                            "user_question": question,
                            "sql_query": pandas_query,
                            "table_info": self.df.info,
                            "query_results": f"Failed to query the database after {max_attempts} attempts. Last error: {error_message}",
                        }
                    )
                    return {
                        "query": pandas_query,
                        "results": None,
                        "answer": answer,
                        "attempts": attempt,
                    }

                question = f"The previous query resulted in an error: '{error_message}'. The query was: '{pandas_query}'. Please correct the query and try again."

        answer = self.reporter_chain.invoke(
            {
                "user_question": question,
                "sql_query": None,
                "table_info": self.df.info,
                "query_results": f"Failed to query the database after {max_attempts} attempts. Provide an excuse message and try to justify. ",
            }
        )
        return {
            "query": pandas_query,
            "results": None,
            "answer": answer,
            "attempts": attempt,
        }


from typing import Literal


class SQLAgent:
    def __init__(self, db_url, model):
        load_dotenv()
        self.api_key = os.getenv("OPENWEBUI_API_KEY")
        self.llm = None
        self.engine = create_engine(url=db_url)
        self.db = SQLDatabase(engine=self.engine, schema="bdschema")
        self.setup_llm(model)
        self.developer_chain = None
        self.question_generator = None
        self.setup_question_generator()
        self.setup_dev_chain()
        self.execute_query = QuerySQLDataBaseTool(db=self.db, verbose=True)
        self.reporter_chain = None
        self.setup_reporter_chain()

    def setup_llm(self, model):
        key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=model, openai_api_key=key, temperature=0)

    def setup_question_generator(self):

        class Kpi(BaseModel):
            kpi: str = Field(description="the kpi")
            title: str = Field(description="a fancy title for the Kpi")
            display_mode: Literal["chart", "number"] = Field(
                description="the display mode of the result, if it is a single number or a list of number (i.e. a chart)"
            )

        class Kpis(BaseModel):
            kpis: List[Kpi] = Field(description="List of KPIs")

        parser = JsonOutputParser(pydantic_object=Kpis)

        self.question_generator_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a data analyst who generates insightful KPIs about databases.
            Given a database schema, generate 10 relevant KPIs from a {theme} point of view and context (deep dive from a previous KPI): {context} , that would help someone 
            understand the data better. For example, if you want a sales POV, consider :
            Total Revenue, 
            Profit, 
            Margin, 
            Sales Growth Rate (from one period to the next), 
            Average Order Value (Average amount spent per transaction),
            Number of sales by the top 10 best salespersons.

            The first questions are the most important ones. 
            You must be sure to find relevant data according to the db schema. \n
            Format your response as : {format_instructions}""",
                ),
                (
                    "user",
                    "Schema:\n{schema}\n\nWrite between 5 to 8 KPIs for data discovery.",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        self.question_generator = self.question_generator_prompt | self.llm | parser

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
        - Write a very very very concise report of the results in natural language, not more than 1 or 2 sentances ! 
        - Don't refer to column or table names, make it fancy for the user.
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
            attempt += 1
            try:
                query = extract_sql_query(
                    self.developer_chain.invoke({"question": ask_the_dev})
                )
                queries.append(query)
                with self.engine.begin() as conn:
                    results = pd.read_sql_query(text(query), conn)

                if not results.empty:
                    # c'est ok
                    answer = self.reporter_chain.invoke(
                        {
                            "user_question": question,
                            "sql_query": query,
                            "table_info": self.db.table_info,
                            "query_results": (
                                results.sample(20).to_string()
                                if len(results) >= 20
                                else results.to_string()
                            ),
                        }
                    )
                    return {
                        "query": query,
                        "results": results,
                        "answer": answer,
                        "attempts": attempt,
                    }
                else:
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
