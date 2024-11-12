import streamlit as st
from src.agents import SQLAgent, PlotGenerator, PandasAgent
from datetime import datetime
import pandas as pd
import os
import json

st.set_page_config(layout="wide")

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []


def initialize_ss_variable(vars: list[str]):
    for var in vars:
        if var not in st.session_state:
            st.session_state[var] = None


def main():
    import base64
    from PIL import Image

    # Function to convert image to base64
    def img_to_base64(img_path):
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    # Custom CSS to center elements and style the title
    st.markdown(
        """
    <style>
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    .title-icon {
        width: 100px;  /* Adjust size as needed */
        height: 100px;
        margin-right: 10px;
    }
    .title-text {
        font-size: 48px;
        font-weight: bold;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Convert image to base64
    img_base64 = img_to_base64("logo.png")

    # Centered title with icon

    # Create a sidebar for navigation
    page = st.sidebar.radio("Navigate", ["Data Discovery", "Agent", "Past Requests"])

    # DATABASE
    db_url = "postgresql://postgres:TH1BD0UX!@localhost:5432/Sport-up"

    if page == "Pandas Agent":
        st.markdown(
            f"""
        <div class="title-container">
            <img src="data:image/png;base64,{img_base64}" class="title-icon">
            <span class="title-text">InsightDB Pandas</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        pandas_agent_page()

    elif page == "Agent":
        st.markdown(
            f"""
        <div class="title-container">
            <img src="data:image/png;base64,{img_base64}" class="title-icon">
            <span class="title-text">InsightDB</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
        main_page(db_url)
    elif page == "Data Discovery":

        st.markdown(
            f"""
        <div class="title-container">
            <img src="data:image/png;base64,{img_base64}" class="title-icon">
            <span class="title-text">DiscoverDB</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.warning(
            "DiscoverDB is a powerful tool to visualize insightful KPIs hidden in your data.",
            icon=":material/signal_cellular_alt:",
        )
        st.markdown(
            "## You are currently connected to the :blue-background[Sport-Up]  database."
        )
        st.info(
            ":blue-background[Sport-Up] is the database of a company selling products all over the world.",
            icon="ℹ️",
        )
        data_discovery_page(db_url)
    elif page == "Past Requests":
        history_page()


def data_discovery_page(db_url):

    sql_agent = SQLAgent(db_url=db_url, model="gpt-4o")
    plot_agent = PlotGenerator(model="gpt-4o")
    if "data_discovery" not in st.session_state:
        st.session_state["data_discovery"] = {}

    theme = st.selectbox(
        "Choose a perspective from which you will analyse the data : ",
        ["Sales", "Marketing"],
    )
    if "discover_data_state" not in st.session_state:
        st.session_state.discover_data_state = None
        st.session_state.should_display = False

    # Main discover button
    discover = st.button("Click to discover :blue-background[Sport-Up] relevant KPIs")

    if discover:
        st.session_state.should_display = True
        st.session_state.discover_data_state = discover_data(
            sql_agent, plot_agent, theme, st.session_state.discover_data_state
        )

    # Always display if should_display is True
    if st.session_state.should_display:
        st.session_state.discover_data_state = discover_data(
            sql_agent, plot_agent, theme, st.session_state.discover_data_state
        )


def discover_data(sql_agent, plot_agent, theme, session_state=None):
    if session_state is None:
        # Initialize the session state if it doesn't exist
        output_q = sql_agent.question_generator.invoke(
            {"schema": sql_agent.db.table_info, "theme": theme, "context": None}
        )
        session_state = {
            "kpis": output_q["kpis"],
            "displayed_kpis": [],
            "counter": 0,
            "output": {},
            "charts_data": {},  # Store chart data for persistence
        }

    # Create columns for layout
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    # Display existing charts first
    for kpi_id, chart_data in session_state["charts_data"].items():
        col = columns[chart_data["position"] % 3]
        exp = col.expander(chart_data["title"], expanded=True)

        if chart_data["display_mode"] == "chart":
            try:
                exp.altair_chart(chart_data["chart"], use_container_width=True)
            except Exception as e:
                st.error(chart_data["error"])
        else:
            exp.markdown(f"## {chart_data['value']}")
            exp.markdown(chart_data["answer"])

        # Add "Related KPIs" button with unique key
        if exp.button(":material/add: related KPIs", key=f"related_kpis_{kpi_id}"):
            related_kpis = sql_agent.question_generator.invoke(
                {
                    "schema": sql_agent.db.table_info,
                    "theme": theme,
                    "context": chart_data["title"],
                }
            )
            session_state["kpis"].extend(related_kpis["kpis"])

    # Process and display new KPIs
    for kpi in session_state["kpis"]:
        if kpi["title"] not in [
            d["title"] for d in session_state["charts_data"].values()
        ]:
            q = kpi["kpi"]
            title = kpi["title"]
            display_mode = kpi["display_mode"]

            with st.spinner(title):
                results = sql_agent.process(question=q, max_attempts=2)

            df = results["results"]
            kpi.update(results)
            session_state["output"].update(kpi)

            if df is not None:
                col = columns[session_state["counter"] % 3]
                exp = col.expander(title, expanded=True)

                with st.spinner(f"Generating {title}..."):
                    chart_data = {
                        "title": title,
                        "display_mode": display_mode,
                        "position": session_state["counter"],
                    }

                    if display_mode == "chart":
                        plotting_results = plot_agent.generate_plot(
                            question=q,
                            df=df,
                            query=results["query"],
                            explanation=results["answer"],
                            max_attempts=2,
                        )
                        chart, error, code = (
                            plotting_results["chart"],
                            plotting_results["error"],
                            plotting_results["code"],
                        )
                        try:
                            exp.altair_chart(chart, use_container_width=True)
                            chart_data.update({"chart": chart, "error": error})
                        except Exception as e:
                            st.error(error)
                            chart_data["error"] = str(e)
                    else:
                        value = round(results["results"].iloc[0, 0], 1)
                        exp.markdown(f"## {value}")
                        exp.markdown(results["answer"])
                        chart_data.update({"value": value, "answer": results["answer"]})

                    # Store chart data for persistence
                    session_state["charts_data"][
                        f'kpi_{session_state["counter"]}'
                    ] = chart_data

                    # Add "Related KPIs" button with unique key
                    if exp.button(
                        ":material/add: related KPIs",
                        key=f"related_kpis_{session_state['counter']}",
                    ):
                        related_kpis = sql_agent.question_generator.invoke(
                            {
                                "schema": sql_agent.db.table_info,
                                "theme": theme,
                                "context": title,
                            }
                        )
                        session_state["kpis"].extend(related_kpis["kpis"])

                session_state["counter"] += 1

    return session_state


# def discover_data(dev_agent, plot_agent, theme):
#     output_q = dev_agent.question_generator.invoke(
#         {"schema": dev_agent.db.table_info, "theme": theme}
#     )
#     kpis = output_q["kpis"]
#     output = {}
#     counter = 0

#     for kpi in kpis:
#         q = kpi["kpi"]
#         title = kpi["title"]
#         display_mode = kpi["display_mode"]
#         with st.spinner(title):
#             results = dev_agent.process(question=q, max_attempts=2)
#         df = results["results"]
#         kpi.update(results)
#         output.update(kpi)

#         if df is not None:
#             if counter % 3 == 0:
#                 # st.divider()
#                 col1, col2, col3 = st.columns(3)
#                 col = col1
#             if counter % 3 == 1:
#                 col = col2
#             elif counter % 3 == 2:
#                 col = col3
#             exp = col.expander(title, expanded=True)
#             with st.spinner(f"Generating {title}..."):
#                 if display_mode == "chart":
#                     plotting_results = plot_agent.generate_plot(
#                         question=q,
#                         df=df,
#                         query=results["query"],
#                         explanation=results["answer"],
#                         max_attempts=2,
#                     )
#                     chart, error, code = (
#                         plotting_results["chart"],
#                         plotting_results["error"],
#                         plotting_results["code"],
#                     )
#                     try:
#                         # box(col=col, title=title, chart=chart)
#                         exp.altair_chart(chart, use_container_width=True)
#                     except Exception as e:
#                         st.error(error)
#                 else:
#                     # box(
#                     #     col=col,
#                     #     title=title,
#                     #     result=round(results["results"].iloc[0, 0], 1),
#                     #     answer=results["answer"],
#                     # )
#                     exp.markdown(f"## {round(results['results'].iloc[0,0], 1)}")
#                     exp.markdown(results["answer"])
#                 exp.button(":material/add: related KPIs")
#         counter += 1


# def box(col, title, result=None, answer=None, chart=None):
#     if chart is not None:
#         col.markdown(
#             f"""
#             <div style="
#                 border: 1px solid #cccccc;  /* Border for a subtle contour */
#                 padding: 15px;
#                 border-radius: 5px;  /* Rounded corners similar to expanders */
#                 width: 300px;  /* Fixed width */
#                 height: 200px; /* Fixed height */
#                 overflow: auto; /* Scrollable if content exceeds size */
#             ">
#                 <p>{title}</p>
#                 <p>{col.altair_chart(chart, use_container_width=True)}</p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#     else:
#         col.markdown(
#             f"""
#             <div style="
#                 border: 1px solid #cccccc;  /* Border for a subtle contour */
#                 padding: 15px;
#                 border-radius: 5px;  /* Rounded corners similar to expanders */
#                 width: 300px;  /* Fixed width */
#                 height: 200px; /* Fixed height */
#                 overflow: auto; /* Scrollable if content exceeds size */
#             ">
#                 <p>{title}</p>
#                 <p>{result}</p>
#                 <p>{answer}</p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )


def data_scientist_process(dev_agent, plot_agent, df_name):
    output_q = dev_agent.question_generator.invoke({"schema": dev_agent.df_schema})
    insights = output_q["insights"]
    output = {}
    counter = 0

    for insight in insights:

        question = insight["precise_query"]
        explanation = insight["explanation"]
        title = insight["title"]
        request = question + f"({explanation})"
        with st.spinner(title):
            results = dev_agent.process(question=request, max_attempts=2)
        df = results["results"]
        answer = results["answer"]
        if df is not None:
            if counter % 3 == 0:
                st.divider()
                col1, col2, col3 = st.columns(3)
                col = col1
            if counter % 3 == 1:
                col = col2
            elif counter % 3 == 2:
                col = col3
            with col.expander(question, expanded=True):

                col.code(results["query"])
                # st.dataframe(df)
                with st.spinner(f"Generating {title}..."):
                    plotting_results = plot_agent.generate_plot(
                        question=question,
                        df=df,
                        query=results["query"],
                        explanation=results["answer"],
                        max_attempts=2,
                    )
                chart, error, code = (
                    plotting_results["chart"],
                    plotting_results["error"],
                    plotting_results["code"],
                )
                try:
                    col.altair_chart(chart, use_container_width=True)
                    chart.save(f"../open_data_soft/{df_name}/{title}.png")
                except Exception as e:
                    st.error(error)
                output[title] = {}
                output[title]["question"] = explanation
                output[title]["answer"] = answer
                with open(f"../open_data_soft/{df_name}/insights.json", "w") as outfile:
                    json.dump(output, outfile)

        counter += 1


def pandas_agent_page():
    dfs_folder = "../open_data_soft/"
    dfs = [f for f in os.listdir(dfs_folder) if f.endswith(".parquet")]
    df_path = st.selectbox("Select the dataframe:", dfs)
    df_name = df_path.split(".")[0]
    if df_name not in st.session_state:
        st.session_state[df_name] = {}
        st.session_state[df_name]["df"] = pd.read_parquet(
            os.path.join(dfs_folder, df_path)
        )
    with st.expander("Data preview", expanded=True):
        st.dataframe(st.session_state[df_name]["df"].head(10))
    ################

    pd_agent = PandasAgent(model="gpt-4o", df=st.session_state[df_name])
    plot_agent = PlotGenerator(model="gpt-4o")

    run_data_scientist = st.button("Run data scientist process")
    if run_data_scientist:
        os.makedirs(f"../open_data_soft/{df_name}", exist_ok=True)
        data_scientist_process(
            dev_agent=pd_agent, plot_agent=plot_agent, df_name=df_name
        )

    question = st.text_input("Enter your question about the dataframe:", value="")
    run_query = st.button("Run Query", use_container_width=True)
    if run_query:
        with st.spinner("Processing your question..."):
            results = pd_agent.process(question=question, max_attempts=5)
            st.code(results["query"])
            st.dataframe(results["results"])
            st.write(results["answer"])
            plotting_results = plot_agent.generate_plot(
                question=question,
                df=results["results"],
                query=results["query"],
                explanation=results["answer"],
                max_attempts=2,
            )
            chart, error, code = (
                plotting_results["chart"],
                plotting_results["error"],
                plotting_results["code"],
            )
            try:
                st.altair_chart(chart, use_container_width=True)
                st.code(code)
            except Exception as e:
                st.error(error)


def main_page(db_url):
    agent = SQLAgent(db_url=db_url, model="gpt-4o")
    plot_agent = PlotGenerator(model="gpt-4o")
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
    # btcol1, btcol2, btcol3 = st.columns(3)
    run_query = st.button(
        "Run Query", disabled=st.session_state.question == "", use_container_width=True
    )
    # plot = btcol2.button(
    #     "Generate a plot",
    #     use_container_width=True,
    #     # disabled=(st.session_state.df is None),
    # )
    # add_to_history = btcol3.button(
    #     "Add to history",
    #     use_container_width=True,
    #     disabled=(st.session_state.explanation is None),
    # )
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
            col1.dataframe(st.session_state.df, use_container_width=True)
        # if plot:
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
            st.code(plotting_results["code"])

    # if add_to_history:
    #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     st.session_state.history.append(
    #         {
    #             "timestamp": timestamp,
    #             "question": st.session_state.question,
    #             "sql_query": st.session_state.query,
    #             "results": st.session_state.df,
    #             "chart": st.session_state.chart,
    #             "answer": st.session_state.explanation,
    #             "attempts": st.session_state.attempts,
    #             "plotting_attempts": st.session_state.plotting_attempts,
    #         }
    #     )

    #     st.success("Results added to history!")


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
