"""Real-time rescheduling and machine failure simulation module."""

from free_claude_code.scheduler.models import ReschedulingComparison
from free_claude_code.scheduler.optimizer import ProductionOptimizer


class ReschedulingSimulator:
    def __init__(self, optimizer: ProductionOptimizer) -> None:
        self.optimizer = optimizer

    def simulate_failure(
        self,
        downed_machines: list[str],
        reschedule_time: float = 0.0,
    ) -> ReschedulingComparison:
        # Run baseline schedule
        before_result = self.optimizer.schedule(
            objective=(self.optimizer.orders
            and self.optimizer.orders[0]
            and "Balanced optimization")
            or "Balanced optimization"
        )

        # Run rescheduled with downed machines excluded
        after_result = self.optimizer.schedule(
            objective=before_result.optimization_objective,
            excluded_machines=downed_machines,
        )

        # Generate impact summary
        makespan_diff = after_result.makespan - before_result.makespan
        late_diff = after_result.late_jobs_count - before_result.late_jobs_count
        impact = (
            f"Machine(s) {', '.join(downed_machines)} marked DOWN at t={reschedule_time}h. "
            f"Makespan changed by {makespan_diff:+.2f}h (Before: {before_result.makespan}h -> After: {after_result.makespan}h). "
            f"Late jobs changed by {late_diff:+d} (Before: {before_result.late_jobs_count} -> After: {after_result.late_jobs_count}). "
            f"New bottleneck machine: {after_result.bottleneck_machine}."
        )

        return ReschedulingComparison(
            before_result=before_result,
            after_result=after_result,
            downed_machines=downed_machines,
            rescheduled_at_time=reschedule_time,
            impact_summary=impact,
        )
