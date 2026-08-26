"""Unit tests for Smart Production Scheduler modules."""


from free_claude_code.scheduler import (
    DataExporter,
    ProcessingTimePredictor,
    ProductionOptimizer,
    ReschedulingSimulator,
    get_sample_historical_data,
    get_sample_machines,
    get_sample_production_orders,
)


def test_sample_data_loading() -> None:
    orders = get_sample_production_orders()
    machines = get_sample_machines()
    historical = get_sample_historical_data()

    assert len(orders) > 0
    assert len(machines) > 0
    assert len(historical) > 0


def test_ml_predictor() -> None:
    historical = get_sample_historical_data()
    predictor = ProcessingTimePredictor()
    res = predictor.train(historical)

    assert res["status"] == "success"
    assert predictor.is_trained is True

    pred = predictor.predict_processing_time(
        part="Gear Shaft A",
        material="Steel 4140",
        operation="Turning & Milling",
        fallback_time=2.5,
    )
    assert isinstance(pred, float)
    assert pred > 0


def test_optimizer_objectives() -> None:
    orders = get_sample_production_orders()
    machines = get_sample_machines()
    optimizer = ProductionOptimizer(orders, machines)

    objectives = [
        "Balanced optimization",
        "Minimum production time",
        "Minimum machine idle time",
        "Minimum late jobs",
        "Minimum changeover time",
    ]

    for obj in objectives:
        result = optimizer.schedule(objective=obj)  # type: ignore
        assert result.makespan > 0
        assert len(result.scheduled_jobs) == len(orders)
        assert result.optimization_objective == obj


def test_rescheduling_simulator() -> None:
    orders = get_sample_production_orders()
    machines = get_sample_machines()
    optimizer = ProductionOptimizer(orders, machines)
    simulator = ReschedulingSimulator(optimizer)

    comparison = simulator.simulate_failure(
        downed_machines=["CNC-01"], reschedule_time=2.0
    )
    assert comparison.before_result.makespan > 0
    assert comparison.after_result.makespan > 0
    assert "CNC-01" in comparison.downed_machines


def test_data_exporter() -> None:
    orders = get_sample_production_orders()
    machines = get_sample_machines()
    optimizer = ProductionOptimizer(orders, machines)
    result = optimizer.schedule(objective="Balanced optimization")

    orders_csv = DataExporter.export_orders_csv(orders)
    machines_csv = DataExporter.export_machines_csv(machines)
    schedule_csv = DataExporter.export_schedule_csv(result)

    assert "Job ID" in orders_csv
    assert "Machine ID" in machines_csv
    assert "Start Time" in schedule_csv
