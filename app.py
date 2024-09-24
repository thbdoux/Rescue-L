import streamlit as st
from src.agents import SQLAgent, DBDescriptor, PlotGenerator
from datetime import datetime

st.set_page_config(layout="wide")

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []


def initialize_ss_variable(vars: list[str]):
    for var in vars:
        if var not in st.session_state:
            st.session_state[var] = None


def main():
    st.title("SQL Agent")

    # Create a sidebar for navigation
    page = st.sidebar.radio("Navigate", ["Agent", "Past Requests", "DB Description"])

    # DATABASE
    db_url = "postgresql://postgres:TH1BD0UX!@localhost:5432/Sport-up"

    if page == "Agent":
        main_page(db_url)
    elif page == "Past Requests":
        history_page()
    elif page == "DB Description":
        db_page(db_url)


def db_page(db_url):
    col1, col2 = st.columns(2)
    descriptor = DBDescriptor(db_url=db_url, model="gpt-4o-mini")
    col1.markdown("### Raw description")
    col1.code(descriptor.db.table_info)
    col2.markdown("### Explain with LLM")
    if col2.button("Let the LLM Agent describe the database"):
        with st.spinner("Scanning database description..."):
            # col2.write(descriptor.describe_database(), flush=True)
            st.session_state.description = descriptor.describe_database()
            col2.write(st.session_state.description, flush=True)
    if st.session_state.description is not None:
        col2.markdown(st.session_state.description)


def main_page(db_url):
    agent = SQLAgent(db_url=db_url, model="gpt-4o-mini")
    plot_agent = PlotGenerator(model="gpt-4o-mini")
    st.session_state.question = st.text_input(
        "Enter your question about the database:",
        value="",
    )
    if "previous_question" not in st.session_state:
        st.session_state.previous_question = ""
    initialize_ss_variable(
        vars=[
            "explanation",
            "df",
            "query",
            "attempts",
            "chart",
            "plotting_code",
            "plotting_error",
            "plotting_attempts",
        ]
    )
    btcol1, btcol2, btcol3 = st.columns(3)
    run_query = btcol1.button(
        "Run Query", disabled=st.session_state.question == "", use_container_width=True
    )
    plot = btcol2.button(
        "Generate a plot",
        use_container_width=True,
        # disabled=(st.session_state.df is None),
    )
    add_to_history = btcol3.button(
        "Add to history",
        use_container_width=True,
        disabled=(st.session_state.explanation is None),
    )
    if (
        run_query or st.session_state.question != st.session_state.previous_question
    ) and not (st.session_state.question == ""):
        with st.spinner("Processing your question..."):
            results = agent.process(question=st.session_state.question, max_attempts=5)

            (
                st.session_state.df,
                st.session_state.query,
                st.session_state.explanation,
                st.session_state.attempts,
            ) = (
                results["results"],
                results["query"],
                results["answer"],
                results["attempts"],
            )
            st.session_state.chart = None
        st.session_state.previous_question = st.session_state.question

    if st.session_state.query is not None:
        with st.expander("SQL Query", expanded=True):
            st.code(st.session_state.query)

    if st.session_state.explanation is not None:
        with st.expander("Explanation", expanded=True):
            st.warning(st.session_state.explanation)

    if st.session_state.df is not None:

        with st.expander("Query Results", expanded=True):
            col1, col2 = st.columns(2)
            col1.dataframe(st.session_state.df)
        if plot:
            with st.spinner("Generating a plot..."):
                plotting_results = plot_agent.generate_plot(
                    question=st.session_state.question,
                    df=st.session_state.df,
                    query=st.session_state.query,
                    explanation=st.session_state.explanation,
                    max_attempts=2,
                )
                (
                    st.session_state.chart,
                    st.session_state.plotting_code,
                    st.session_state.plotting_error,
                    st.session_state.plotting_attempts,
                ) = (
                    plotting_results["chart"],
                    plotting_results["error"],
                    plotting_results["code"],
                    plotting_results["attempts"],
                )

        if st.session_state.chart is not None:
            col2.altair_chart(st.session_state.chart, use_container_width=True)
            # st.code(st.session_state.plotting_code)

    if add_to_history:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.history.append(
            {
                "timestamp": timestamp,
                "question": st.session_state.question,
                "sql_query": st.session_state.query,
                "results": st.session_state.df,
                "chart": st.session_state.chart,
                "answer": st.session_state.explanation,
                "attempts": st.session_state.attempts,
                "plotting_attempts": st.session_state.plotting_attempts,
            }
        )

        st.success("Results added to history!")


def history_page():
    st.header("Query History")

    if not st.session_state.history:
        st.info("No queries in history yet. Try asking a question on the main page!")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(
                f"Query {len(st.session_state.history) - i}: {entry['question']} ({entry['timestamp']})"
            ):
                st.subheader("Question")
                st.write(entry["question"])

                st.subheader("SQL Query")
                st.code(entry["sql_query"], language="sql")

                st.subheader("Results")
                col1, col2 = st.columns(2)
                col1.dataframe(entry["results"], hide_index=True)
                if entry["chart"] is not None:
                    col2.altair_chart(entry["chart"])

                st.subheader("Answer")
                st.warning(entry["answer"])

                st.error(f"Result reached after {entry['attempts']} attempts.")


if __name__ == "__main__":
    main()
