from abc import ABC, abstractmethod

class BaseHealthCheck(ABC):
    """Abstract class for all data health checks."""

    @abstractmethod
    def run(self, df) -> "HealthCheckResult":
        """Run the health check and return a HealthCheckResult."""
        raise NotImplementedError


class HealthCheckResult:
    """Holds the output of a health check."""

    def __init__(self, name: str, status: str, details: dict):
        self.name = name
        self.status = status  # 'healthy' | 'warning' | 'critical'
        self.details = details

    def __repr__(self):
        return f"{self.name} ({self.status}) -> {self.details}"


class HealthReport:
    """Collect results from multiple health checks."""

    def __init__(self):
        self.results = []

    def add(self, result: HealthCheckResult):
        self.results.append(result)

    def to_dict(self):
        return {
            "checks": [
                {"name": r.name, "status": r.status, "details": r.details}
                for r in self.results
            ]
        }

    def __repr__(self):
        return "\n".join([repr(r) for r in self.results])


class HealthValidator:
    """Coordinator to run multiple health checks."""

    def __init__(self, checks: list):
        self.checks = checks

    def run(self, df) -> HealthReport:
        report = HealthReport()
        for check in self.checks:
            result = check.run(df)
            report.add(result)
        return report
