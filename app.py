import streamlit as st
from src.agents import SQLAgent, DBDescriptor, PlotGenerator
from datetime import datetime

st.set_page_config(layout="wide")

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []


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
    question = st.text_input(
        "Enter your question about the database:",
        value="""
What is the evolution of the sales over the years, for men and women sales separately?
""",
    )

    if st.button("Run Query", use_container_width=True):
        with st.spinner("Processing your question..."):
            results = agent.process(question=question, max_attempts=5)
            df, query, explanation, attempts = (
                results["results"],
                results["query"],
                results["answer"],
                results["attempts"],
            )

            if query is not None:
                with st.expander("SQL Query", expanded=True):
                    st.code(query)

            with st.expander("Explanation", expanded=True):
                st.warning(explanation)

            if df is not None:
                plotting_results = plot_agent.generate_plot(
                    question=question,
                    df=df,
                    query=query,
                    explanation=explanation,
                    max_attempts=5,
                )
                with st.expander("Query Results", expanded=True):
                    col1, col2 = st.columns(2)
                    col1.dataframe(df, hide_index=True)
                    if plotting_results["chart"] is not None:
                        # st.error(
                        #     f"Plotted after {plotting_results['attempts']} attempts"
                        # )
                        col2.altair_chart(
                            plotting_results["chart"], use_container_width=True
                        )
                    else:
                        st.code(plotting_results["code"])

            # st.error(f"Result reached after {results['attempts']+1} attempts.")

            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.append(
                {
                    "timestamp": timestamp,
                    "question": question,
                    "sql_query": query,
                    "results": df,
                    "chart": plotting_results["chart"],
                    "answer": explanation,
                    "attempts": attempts + 1,
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
