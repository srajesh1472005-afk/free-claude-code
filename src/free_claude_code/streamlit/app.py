"""Streamlit web application for AI-Based Smart Production Scheduling and Real-Time Rescheduling System."""

import pandas as pd
import plotly.express as px
import streamlit as st

from free_claude_code.scheduler import (
    DataExporter,
    HistoricalDataRecord,
    MachineInfo,
    ProcessingTimePredictor,
    ProductionOptimizer,
    ProductionOrder,
    ReschedulingSimulator,
    get_sample_historical_data,
    get_sample_machines,
    get_sample_production_orders,
)

st.set_page_config(
    page_title="AI Smart Production Scheduler & Rescheduler",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for professional manufacturing dashboard
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state() -> None:
    if "orders" not in st.session_state:
        st.session_state.orders = get_sample_production_orders()
    if "machines" not in st.session_state:
        st.session_state.machines = get_sample_machines()
    if "historical" not in st.session_state:
        st.session_state.historical = get_sample_historical_data()
    if "predictor" not in st.session_state:
        st.session_state.predictor = ProcessingTimePredictor()
    if "trained_ml" not in st.session_state:
        st.session_state.trained_ml = False


initialize_session_state()

# Sidebar Navigation
st.sidebar.title("🏭 Smart Scheduler")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Dashboard",
        "📂 Data Inputs & Uploads",
        "🤖 ML Processing Time Predictor",
        "⚡ Optimization & Scheduling Engine",
        "📈 Interactive Visualizations",
        "🚨 Real-Time Failure & Rescheduling",
        "📥 Reports & CSV Exports",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status**: Online\n\n"
    "**Engine**: Python / Scikit-Learn / Plotly / Streamlit\n"
    "**Version**: 5.13.11"
)

# 1. Executive Dashboard
if page == "📊 Executive Dashboard":
    st.markdown(
        '<div class="main-header">AI-Based Smart Production Scheduling & Real-Time Rescheduling</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Real-time manufacturing optimization, machine learning estimation, and failure recovery.</div>',
        unsafe_allow_html=True,
    )

    # Quick run baseline to display summary metrics
    optimizer = ProductionOptimizer(
        st.session_state.orders,
        st.session_state.machines,
        st.session_state.predictor if st.session_state.trained_ml else None,
    )
    result = optimizer.schedule(objective="Balanced optimization")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Makespan", f"{result.makespan} hrs", delta=None)
    with col2:
        st.metric("Total Processing", f"{result.total_processing_time} hrs")
    with col3:
        st.metric("Total Changeover", f"{result.total_changeover_time} hrs")
    with col4:
        st.metric(
            "Late Jobs",
            f"{result.late_jobs_count}",
            delta="Warning" if result.late_jobs_count > 0 else "Optimal",
            delta_color="inverse" if result.late_jobs_count > 0 else "normal",
        )
    with col5:
        st.metric("Bottleneck Machine", f"{result.bottleneck_machine}")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("⚡ Machine Utilization Overview")
        util_df = pd.DataFrame(
            list(result.machine_utilization.items()),
            columns=["Machine ID", "Utilization (%)"],
        )
        fig_util = px.bar(
            util_df,
            x="Machine ID",
            y="Utilization (%)",
            color="Utilization (%)",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_util.update_layout(yaxis_range=[0, 100], margin={"l": 20, "r": 20, "t": 30, "b": 20})
        st.plotly_chart(fig_util, use_container_width=True)

    with col_b:
        st.subheader("💡 AI Production Recommendations")
        recommendations = []
        if result.late_jobs_count > 0:
            recommendations.append(
                f"⚠️ **Due Date Risk**: {result.late_jobs_count} job(s) exceed due dates. Consider prioritizing 'Minimum late jobs' objective or reallocating quantities."
            )
        if result.bottleneck_machine != "None":
            max_load = result.machine_workload.get(result.bottleneck_machine, 0)
            recommendations.append(
                f"🔧 **Bottleneck Identified**: Machine `{result.bottleneck_machine}` carries the highest workload ({max_load} hrs). Consider load balancing."
            )
        if not st.session_state.trained_ml:
            recommendations.append(
                "🤖 **Machine Learning**: Historical data is not trained. Train the RandomForestRegressor in the ML tab to improve prediction accuracy."
            )
        if not recommendations:
            recommendations.append(
                "✅ All systems operating within optimal parameters. Schedule is balanced and on track."
            )

        for rec in recommendations:
            st.markdown(f"- {rec}")

    st.markdown("---")
    st.subheader("📋 Scheduled Production Summary")
    summary_df = pd.DataFrame(
        [
            {
                "Job ID": j.job_id,
                "Part": j.part,
                "Machine": j.machine_id,
                "Qty": j.quantity,
                "Start (h)": j.start_time,
                "Finish (h)": j.finish_time,
                "Due (h)": j.due_date,
                "Late?": "YES" if j.is_late else "NO",
            }
            for j in result.scheduled_jobs
        ]
    )
    st.dataframe(summary_df, use_container_width=True)

# 2. Data Inputs & Uploads
elif page == "📂 Data Inputs & Uploads":
    st.markdown(
        '<div class="main-header">Production Orders & Machine Information</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Manage production orders, machine status, and historical data via CSV upload or manual entry.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Production Orders", "Machine Information", "Historical Manufacturing Data"]
    )

    with tab1:
        st.subheader("Production Orders")
        col_up, col_btn = st.columns([3, 1])
        with col_up:
            order_file = st.file_uploader(
                "Upload Production Orders CSV", type=["csv"], key="order_csv"
            )
        with col_btn:
            if st.button("Load Sample Orders"):
                st.session_state.orders = get_sample_production_orders()
                st.success("Sample orders loaded!")

        if order_file is not None:
            try:
                df_in = pd.read_csv(order_file)
                # Parse into ProductionOrder objects
                new_orders = []
                for _, row in df_in.iterrows():
                    new_orders.append(
                        ProductionOrder(
                            job_id=str(row.get("Job ID", f"JOB-{_}")),
                            part=str(row.get("Part", "Part A")),
                            machine_id=str(row.get("Machine", "CNC-01")),
                            quantity=int(row.get("Quantity", 10)),
                            material=str(row.get("Material", "Steel")),
                            operation=str(row.get("Operation", "Milling")),
                            processing_time=float(row.get("Processing Time", 2.0)),
                            setup_time=float(row.get("Setup Time", 0.5)),
                            priority=int(row.get("Priority", 3)),
                            due_date=float(row.get("Due Date", 24.0)),
                        )
                    )
                st.session_state.orders = new_orders
                st.success(
                    f"Successfully loaded {len(new_orders)} production orders from CSV."
                )
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")

        orders_df = pd.DataFrame(
            [
                {
                    "Job ID": o.job_id,
                    "Part": o.part,
                    "Machine": o.machine_id,
                    "Quantity": o.quantity,
                    "Material": o.material,
                    "Operation": o.operation,
                    "Processing Time": o.processing_time,
                    "Setup Time": o.setup_time,
                    "Priority": o.priority,
                    "Due Date": o.due_date,
                }
                for o in st.session_state.orders
            ]
        )
        edited_orders_df = st.data_editor(
            orders_df, num_rows="dynamic", use_container_width=True
        )
        if st.button("Save Orders Edits"):
            updated_orders = []
            for _, r in edited_orders_df.iterrows():
                updated_orders.append(
                    ProductionOrder(
                        job_id=str(r["Job ID"]),
                        part=str(r["Part"]),
                        machine_id=str(r["Machine"]),
                        quantity=int(r["Quantity"]),
                        material=str(r["Material"]),
                        operation=str(r["Operation"]),
                        processing_time=float(r["Processing Time"]),
                        setup_time=float(r["Setup Time"]),
                        priority=int(r["Priority"]),
                        due_date=float(r["Due Date"]),
                    )
                )
            st.session_state.orders = updated_orders
            st.success("Production orders updated successfully!")

    with tab2:
        st.subheader("Machine Information & Status")
        col_up_m, col_btn_m = st.columns([3, 1])
        with col_up_m:
            machine_file = st.file_uploader(
                "Upload Machine Info CSV", type=["csv"], key="machine_csv"
            )
        with col_btn_m:
            if st.button("Load Sample Machines"):
                st.session_state.machines = get_sample_machines()
                st.success("Sample machines loaded!")

        if machine_file is not None:
            try:
                df_m = pd.read_csv(machine_file)
                new_machines = []
                for _, row in df_m.iterrows():
                    new_machines.append(
                        MachineInfo(
                            machine_id=str(row.get("Machine ID", "CNC-01")),
                            machine_type=str(row.get("Machine Type", "CNC")),
                            status=str(row.get("Status", "AVAILABLE")),
                            current_job=str(row.get("Current Job", "Idle")),
                            speed_factor=float(row.get("Speed Factor", 1.0)),
                        )
                    )
                st.session_state.machines = new_machines
                st.success(f"Successfully loaded {len(new_machines)} machines.")
            except Exception as e:
                st.error(f"Error parsing machine CSV: {e}")

        machines_df = pd.DataFrame(
            [
                {
                    "Machine ID": m.machine_id,
                    "Machine Type": m.machine_type,
                    "Status": m.status,
                    "Current Job": m.current_job,
                    "Speed Factor": m.speed_factor,
                }
                for m in st.session_state.machines
            ]
        )
        edited_machines_df = st.data_editor(
            machines_df, num_rows="dynamic", use_container_width=True
        )
        if st.button("Save Machine Edits"):
            updated_machines = []
            for _, r in edited_machines_df.iterrows():
                updated_machines.append(
                    MachineInfo(
                        machine_id=str(r["Machine ID"]),
                        machine_type=str(r["Machine Type"]),
                        status=str(r["Status"]),
                        current_job=str(r["Current Job"]),
                        speed_factor=float(r["Speed Factor"]),
                    )
                )
            st.session_state.machines = updated_machines
            st.success("Machine information updated successfully!")

    with tab3:
        st.subheader("Historical Manufacturing Data (for ML Training)")
        col_up_h, col_btn_h = st.columns([3, 1])
        with col_up_h:
            hist_file = st.file_uploader(
                "Upload Historical Data CSV", type=["csv"], key="hist_csv"
            )
        with col_btn_h:
            if st.button("Load Sample Historical Data"):
                st.session_state.historical = get_sample_historical_data()
                st.success("Sample historical data loaded!")

        if hist_file is not None:
            try:
                df_h = pd.read_csv(hist_file)
                new_hist = []
                for _, row in df_h.iterrows():
                    new_hist.append(
                        HistoricalDataRecord(
                            part=str(row.get("Part", "Part A")),
                            material=str(row.get("Material", "Steel")),
                            operation=str(row.get("Operation", "Milling")),
                            cutting_speed=float(row.get("Cutting Speed", 250.0)),
                            feed_rate=float(row.get("Feed Rate", 0.25)),
                            depth_of_cut=float(row.get("Depth of Cut", 2.0)),
                            tool_condition=float(row.get("Tool Condition", 0.9)),
                            actual_processing_time=float(
                                row.get("Actual Processing Time", 2.0)
                            ),
                        )
                    )
                st.session_state.historical = new_hist
                st.success(f"Successfully loaded {len(new_hist)} historical records.")
            except Exception as e:
                st.error(f"Error parsing historical CSV: {e}")

        hist_df = pd.DataFrame(
            [
                {
                    "Part": h.part,
                    "Material": h.material,
                    "Operation": h.operation,
                    "Cutting Speed": h.cutting_speed,
                    "Feed Rate": h.feed_rate,
                    "Depth of Cut": h.depth_of_cut,
                    "Tool Condition": h.tool_condition,
                    "Actual Processing Time": h.actual_processing_time,
                }
                for h in st.session_state.historical
            ]
        )
        edited_hist_df = st.data_editor(
            hist_df, num_rows="dynamic", use_container_width=True
        )
        if st.button("Save Historical Data Edits"):
            updated_hist = []
            for _, r in edited_hist_df.iterrows():
                updated_hist.append(
                    HistoricalDataRecord(
                        part=str(r["Part"]),
                        material=str(r["Material"]),
                        operation=str(r["Operation"]),
                        cutting_speed=float(r["Cutting Speed"]),
                        feed_rate=float(r["Feed Rate"]),
                        depth_of_cut=float(r["Depth of Cut"]),
                        tool_condition=float(r["Tool Condition"]),
                        actual_processing_time=float(r["Actual Processing Time"]),
                    )
                )
            st.session_state.historical = updated_hist
            st.success("Historical data updated successfully!")

# 3. ML Processing Time Predictor
elif page == "🤖 ML Processing Time Predictor":
    st.markdown(
        '<div class="main-header">Machine Learning Processing Time Predictor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Train a RandomForestRegressor using historical manufacturing parameters to predict precise processing times.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Model Training")
        st.write(
            f"Available historical records: **{len(st.session_state.historical)}**"
        )
        if st.button("Train RandomForest Model", type="primary"):
            with st.spinner("Training Random Forest Regressor..."):
                predictor = ProcessingTimePredictor()
                res = predictor.train(st.session_state.historical)
                st.session_state.predictor = predictor
                st.session_state.trained_ml = True
                if res["status"] == "success":
                    st.success("Model trained successfully!")
                else:
                    st.warning(
                        "Insufficient historical data for robust training. Fallback times will be used."
                    )

        if st.session_state.trained_ml:
            st.metric(
                "Mean Absolute Error (MAE)", f"{st.session_state.predictor.mae:.4f} hrs"
            )
            st.metric("R² Score", f"{st.session_state.predictor.r2:.4f}")
        else:
            st.info("Model not trained yet.")

    with col2:
        st.subheader("Feature Importance Analysis")
        if (
            st.session_state.trained_ml
            and st.session_state.predictor.feature_importances
        ):
            imp_dict = st.session_state.predictor.feature_importances
            imp_df = pd.DataFrame(
                list(imp_dict.items()), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Random Forest Feature Importances",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Train the model to visualize feature importances.")

# 4. Optimization & Scheduling Engine
elif page == "⚡ Optimization & Scheduling Engine":
    st.markdown(
        '<div class="main-header">Optimization & Scheduling Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Select your strategic manufacturing objective and generate the optimized production schedule.</div>',
        unsafe_allow_html=True,
    )

    objective = st.selectbox(
        "Select Optimization Objective",
        [
            "Balanced optimization",
            "Minimum production time",
            "Minimum machine idle time",
            "Minimum late jobs",
            "Minimum changeover time",
        ],
    )

    if st.button("Run Optimizer", type="primary", use_container_width=True):
        with st.spinner("Running scheduling optimization..."):
            optimizer = ProductionOptimizer(
                st.session_state.orders,
                st.session_state.machines,
                st.session_state.predictor if st.session_state.trained_ml else None,
            )
            result = optimizer.schedule(objective=objective)
            st.session_state.last_schedule_result = result
            st.success("Optimization completed successfully!")

    if "last_schedule_result" in st.session_state:
        res = st.session_state.last_schedule_result
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Makespan", f"{res.makespan} hrs")
        with col2:
            st.metric("Total Changeover", f"{res.total_changeover_time} hrs")
        with col3:
            st.metric("Late Jobs", f"{res.late_jobs_count}")
        with col4:
            st.metric("Bottleneck", f"{res.bottleneck_machine}")

        st.subheader("Optimized Job Schedule")
        sched_df = pd.DataFrame(
            [
                {
                    "Job ID": j.job_id,
                    "Part": j.part,
                    "Machine": j.machine_id,
                    "Qty": j.quantity,
                    "Setup (h)": j.setup_time,
                    "Proc (h)": j.predicted_processing_time,
                    "Start (h)": j.start_time,
                    "Finish (h)": j.finish_time,
                    "Due (h)": j.due_date,
                    "Late?": "YES" if j.is_late else "NO",
                    "Lateness (h)": j.lateness,
                }
                for j in res.scheduled_jobs
            ]
        )
        st.dataframe(sched_df, use_container_width=True)

# 5. Interactive Visualizations
elif page == "📈 Interactive Visualizations":
    st.markdown(
        '<div class="main-header">Production Visualizations & Analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Interactive Gantt charts, workload distribution, and machine utilization analytics.</div>',
        unsafe_allow_html=True,
    )

    optimizer = ProductionOptimizer(
        st.session_state.orders,
        st.session_state.machines,
        st.session_state.predictor if st.session_state.trained_ml else None,
    )
    result = optimizer.schedule(objective="Balanced optimization")

    if not result.scheduled_jobs:
        st.warning("No scheduled jobs found. Please check your production orders.")
    else:
        st.subheader("Production Gantt Chart (Machine Timelines)")
        gantt_data = [
            {
                "Task": f"{j.job_id} ({j.part})",
                "Machine": j.machine_id,
                "Start": j.start_time,
                "Finish": j.finish_time,
                "Part": j.part,
            }
            for j in result.scheduled_jobs
        ]
        df_gantt = pd.DataFrame(gantt_data)
        fig_gantt = px.timeline(
            df_gantt,
            x_start="Start",
            x_end="Finish",
            y="Machine",
            color="Part",
            title="Machine-wise Job Timelines (Gantt)",
            labels={"Start": "Start Time (hours)", "Finish": "Finish Time (hours)"},
        )
        fig_gantt.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_gantt, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Machine Workload Distribution")
            workload_df = pd.DataFrame(
                list(result.machine_workload.items()),
                columns=["Machine ID", "Workload (hrs)"],
            )
            fig_workload = px.bar(
                workload_df,
                x="Machine ID",
                y="Workload (hrs)",
                color="Workload (hrs)",
                color_continuous_scale="Blues",
                text_auto=True,
            )
            st.plotly_chart(fig_workload, use_container_width=True)

        with col2:
            st.subheader("Machine Utilization (%)")
            util_df = pd.DataFrame(
                list(result.machine_utilization.items()),
                columns=["Machine ID", "Utilization (%)"],
            )
            fig_util = px.bar(
                util_df,
                x="Machine ID",
                y="Utilization (%)",
                color="Utilization (%)",
                color_continuous_scale="Greens",
                text_auto=True,
            )
            fig_util.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig_util, use_container_width=True)

# 6. Real-Time Failure & Rescheduling
elif page == "🚨 Real-Time Failure & Rescheduling":
    st.markdown(
        '<div class="main-header">Real-Time Machine Failure Simulation & Rescheduling</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Simulate unexpected machine breakdowns, automatically exclude downed machines, and view before-and-after KPI comparisons.</div>',
        unsafe_allow_html=True,
    )

    machine_ids = [m.machine_id for m in st.session_state.machines]
    downed_selection = st.multiselect("Select Machine(s) to Simulate DOWN", machine_ids)
    reschedule_time = st.slider("Failure Time (hours from start)", 0.0, 24.0, 2.0, 0.5)

    if st.button(
        "Trigger Failure & Reschedule", type="primary", use_container_width=True
    ):
        optimizer = ProductionOptimizer(
            st.session_state.orders,
            st.session_state.machines,
            st.session_state.predictor if st.session_state.trained_ml else None,
        )
        simulator = ReschedulingSimulator(optimizer)
        comparison = simulator.simulate_failure(
            downed_machines=downed_selection, reschedule_time=reschedule_time
        )
        st.session_state.last_reschedule_comparison = comparison
        st.success("Rescheduling simulation completed!")

    if "last_reschedule_comparison" in st.session_state:
        comp = st.session_state.last_reschedule_comparison
        st.markdown("---")
        st.info(comp.impact_summary)

        st.subheader("Before & After KPI Comparison")
        b = comp.before_result
        a = comp.after_result

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Makespan",
                f"{a.makespan} hrs",
                delta=f"{a.makespan - b.makespan:+.2f} hrs",
            )
        with col2:
            st.metric(
                "Total Changeover",
                f"{a.total_changeover_time} hrs",
                delta=f"{a.total_changeover_time - b.total_changeover_time:+.2f} hrs",
            )
        with col3:
            st.metric(
                "Late Jobs",
                f"{a.late_jobs_count}",
                delta=f"{a.late_jobs_count - b.late_jobs_count:+d}",
            )

        col_b, col_a_sched = st.columns(2)
        with col_b:
            st.markdown("**Baseline Schedule**")
            base_df = pd.DataFrame(
                [
                    {
                        "Job": j.job_id,
                        "Machine": j.machine_id,
                        "Finish": j.finish_time,
                        "Late": j.is_late,
                    }
                    for j in b.scheduled_jobs
                ]
            )
            st.dataframe(base_df, use_container_width=True)

        with col_a_sched:
            st.markdown("**Rescheduled After Failure**")
            after_df = pd.DataFrame(
                [
                    {
                        "Job": j.job_id,
                        "Machine": j.machine_id,
                        "Finish": j.finish_time,
                        "Late": j.is_late,
                    }
                    for j in a.scheduled_jobs
                ]
            )
            st.dataframe(after_df, use_container_width=True)

# 7. Reports & CSV Exports
elif page == "📥 Reports & CSV Exports":
    st.markdown(
        '<div class="main-header">Reports & Data Exports</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Download production inputs, optimized schedules, rescheduled plans, and KPI comparison reports as CSV files.</div>',
        unsafe_allow_html=True,
    )

    optimizer = ProductionOptimizer(
        st.session_state.orders,
        st.session_state.machines,
        st.session_state.predictor if st.session_state.trained_ml else None,
    )
    result = optimizer.schedule(objective="Balanced optimization")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. User Production Input")
        orders_csv = DataExporter.export_orders_csv(st.session_state.orders)
        st.download_button(
            label="Download User Production Input.csv",
            data=orders_csv,
            file_name="User Production Input.csv",
            mime="text/csv",
        )

        st.subheader("2. Machine Status")
        machines_csv = DataExporter.export_machines_csv(st.session_state.machines)
        st.download_button(
            label="Download Machine Status.csv",
            data=machines_csv,
            file_name="Machine Status.csv",
            mime="text/csv",
        )

    with col2:
        st.subheader("3. Optimized Production Schedule")
        sched_csv = DataExporter.export_schedule_csv(result)
        st.download_button(
            label="Download Optimized Production Schedule.csv",
            data=sched_csv,
            file_name="Optimized Production Schedule.csv",
            mime="text/csv",
        )

        if "last_reschedule_comparison" in st.session_state:
            st.subheader("4. KPI Before & After Failure")
            kpi_csv = DataExporter.export_kpi_comparison_csv(
                st.session_state.last_reschedule_comparison
            )
            st.download_button(
                label="Download KPI Before After Failure.csv",
                data=kpi_csv,
                file_name="KPI Before After Failure.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "Run a failure simulation in the previous tab to enable KPI Before After Failure export."
            )
