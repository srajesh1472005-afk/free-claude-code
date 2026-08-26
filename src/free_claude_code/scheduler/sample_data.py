"""Built-in sample datasets for production orders, machines, and historical data."""

import pandas as pd

from free_claude_code.scheduler.models import (
    HistoricalDataRecord,
    MachineInfo,
    ProductionOrder,
)


def get_sample_production_orders() -> list[ProductionOrder]:
    return [
        ProductionOrder(
            job_id="JOB-101",
            part="Gear Shaft A",
            machine_id="CNC-01",
            quantity=50,
            material="Steel 4140",
            operation="Turning & Milling",
            processing_time=2.5,
            setup_time=0.5,
            priority=4,
            due_date=24.0,
        ),
        ProductionOrder(
            job_id="JOB-102",
            part="Housing Cover B",
            machine_id="CNC-02",
            quantity=30,
            material="Aluminum 6061",
            operation="Boring",
            processing_time=1.8,
            setup_time=0.4,
            priority=5,
            due_date=16.0,
        ),
        ProductionOrder(
            job_id="JOB-103",
            part="Drive Axle C",
            machine_id="CNC-01",
            quantity=40,
            material="Steel 4340",
            operation="Lathe Cut",
            processing_time=3.0,
            setup_time=0.6,
            priority=3,
            due_date=36.0,
        ),
        ProductionOrder(
            job_id="JOB-104",
            part="Flange Plate D",
            machine_id="CNC-03",
            quantity=60,
            material="Stainless Steel",
            operation="Surface Grinding",
            processing_time=1.2,
            setup_time=0.3,
            priority=2,
            due_date=30.0,
        ),
        ProductionOrder(
            job_id="JOB-105",
            part="Bracket E",
            machine_id="CNC-02",
            quantity=25,
            material="Aluminum 7075",
            operation="Milling",
            processing_time=2.0,
            setup_time=0.5,
            priority=4,
            due_date=20.0,
        ),
        ProductionOrder(
            job_id="JOB-106",
            part="Spindle Shaft F",
            machine_id="CNC-03",
            quantity=35,
            material="Steel 4140",
            operation="Turning",
            processing_time=2.7,
            setup_time=0.4,
            priority=3,
            due_date=40.0,
        ),
    ]


def get_sample_machines() -> list[MachineInfo]:
    return [
        MachineInfo(
            machine_id="CNC-01",
            machine_type="5-Axis CNC Lathe",
            status="AVAILABLE",
            current_job="Idle",
            speed_factor=1.0,
        ),
        MachineInfo(
            machine_id="CNC-02",
            machine_type="Vertical Machining Center",
            status="AVAILABLE",
            current_job="Idle",
            speed_factor=1.0,
        ),
        MachineInfo(
            machine_id="CNC-03",
            machine_type="Horizontal Mill",
            status="AVAILABLE",
            current_job="Idle",
            speed_factor=1.0,
        ),
    ]


def get_sample_historical_data() -> list[HistoricalDataRecord]:
    return [
        HistoricalDataRecord(
            part="Gear Shaft A",
            material="Steel 4140",
            operation="Turning & Milling",
            cutting_speed=250.0,
            feed_rate=0.25,
            depth_of_cut=2.0,
            tool_condition=0.85,
            actual_processing_time=2.45,
        ),
        HistoricalDataRecord(
            part="Housing Cover B",
            material="Aluminum 6061",
            operation="Boring",
            cutting_speed=400.0,
            feed_rate=0.35,
            depth_of_cut=1.5,
            tool_condition=0.90,
            actual_processing_time=1.75,
        ),
        HistoricalDataRecord(
            part="Drive Axle C",
            material="Steel 4340",
            operation="Lathe Cut",
            cutting_speed=220.0,
            feed_rate=0.20,
            depth_of_cut=2.5,
            tool_condition=0.75,
            actual_processing_time=3.10,
        ),
        HistoricalDataRecord(
            part="Flange Plate D",
            material="Stainless Steel",
            operation="Surface Grinding",
            cutting_speed=180.0,
            feed_rate=0.15,
            depth_of_cut=1.0,
            tool_condition=0.95,
            actual_processing_time=1.20,
        ),
        HistoricalDataRecord(
            part="Bracket E",
            material="Aluminum 7075",
            operation="Milling",
            cutting_speed=380.0,
            feed_rate=0.30,
            depth_of_cut=1.8,
            tool_condition=0.80,
            actual_processing_time=1.95,
        ),
        HistoricalDataRecord(
            part="Spindle Shaft F",
            material="Steel 4140",
            operation="Turning",
            cutting_speed=240.0,
            feed_rate=0.22,
            depth_of_cut=2.2,
            tool_condition=0.88,
            actual_processing_time=2.65,
        ),
        HistoricalDataRecord(
            part="Gear Shaft A",
            material="Steel 4140",
            operation="Turning & Milling",
            cutting_speed=260.0,
            feed_rate=0.26,
            depth_of_cut=2.0,
            tool_condition=0.60,
            actual_processing_time=2.70,
        ),
        HistoricalDataRecord(
            part="Housing Cover B",
            material="Aluminum 6061",
            operation="Boring",
            cutting_speed=410.0,
            feed_rate=0.36,
            depth_of_cut=1.5,
            tool_condition=0.50,
            actual_processing_time=1.90,
        ),
    ]


def orders_to_dataframe(orders: list[ProductionOrder]) -> pd.DataFrame:
    return pd.DataFrame(
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
            for o in orders
        ]
    )


def machines_to_dataframe(machines: list[MachineInfo]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Machine ID": m.machine_id,
                "Machine Type": m.machine_type,
                "Status": m.status,
                "Current Job": m.current_job,
                "Speed Factor": m.speed_factor,
            }
            for m in machines
        ]
    )


def historical_to_dataframe(records: list[HistoricalDataRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Part": r.part,
                "Material": r.material,
                "Operation": r.operation,
                "Cutting Speed": r.cutting_speed,
                "Feed Rate": r.feed_rate,
                "Depth of Cut": r.depth_of_cut,
                "Tool Condition": r.tool_condition,
                "Actual Processing Time": r.actual_processing_time,
            }
            for r in records
        ]
    )
