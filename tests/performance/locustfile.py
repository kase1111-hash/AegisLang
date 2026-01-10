"""
AegisLang Performance Testing with Locust.

This module provides load testing scenarios for the AegisLang API.

Usage:
    # Install locust
    pip install locust

    # Run locally
    locust -f tests/performance/locustfile.py --host http://localhost:8080

    # Run headless with specific parameters
    locust -f tests/performance/locustfile.py \
        --host http://localhost:8080 \
        --headless \
        --users 100 \
        --spawn-rate 10 \
        --run-time 60s

    # Run with web UI
    locust -f tests/performance/locustfile.py --host http://localhost:8080
    # Then open http://localhost:8089
"""

import os
import json
import random
from typing import Optional
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_POLICIES = [
    """# Anti-Money Laundering Policy

## Customer Due Diligence
Financial institutions must verify customer identity.
Banks shall perform enhanced due diligence for high-risk customers.
Institutions are required to maintain records for 5 years.

## Transaction Monitoring
Organizations must report suspicious transactions exceeding $10,000.
Employees shall not process transactions without authorization.
""",
    """# Data Protection Policy

## Data Collection
Organizations must obtain consent before collecting personal data.
Companies shall document the purpose of data collection.

## Data Subject Rights
Individuals may request access to their personal data.
Users are permitted to request deletion of their data.
""",
    """# Employee Conduct Policy

## Workplace Standards
Employees must adhere to the company dress code.
Staff shall report to work on time.

## Prohibited Actions
Employees shall not engage in harassment.
Staff must not share confidential information.
""",
]


# =============================================================================
# Utility Functions
# =============================================================================

def get_sample_policy() -> str:
    """Get a random sample policy."""
    return random.choice(SAMPLE_POLICIES)


# =============================================================================
# User Behaviors
# =============================================================================

class AegisLangUser(HttpUser):
    """
    Simulated user behavior for AegisLang API.
    """

    # Wait between 1-5 seconds between tasks
    wait_time = between(1, 5)

    def on_start(self):
        """Called when a user starts."""
        self.doc_ids = []

    @task(10)
    def health_check(self):
        """
        High-frequency health check requests.
        Weight: 10 (most common operation)
        """
        with self.client.get(
            "/api/v1/health",
            name="Health Check",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") not in ["healthy", "degraded"]:
                    response.failure("Unexpected health status")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def list_documents(self):
        """
        List all documents.
        Weight: 5 (common operation)
        """
        with self.client.get(
            "/api/v1/documents",
            name="List Documents",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def ingest_document(self):
        """
        Ingest a new policy document.
        Weight: 3 (less common, resource intensive)
        """
        policy = get_sample_policy()
        files = {
            "file": (
                f"policy_{random.randint(1000, 9999)}.md",
                policy.encode("utf-8"),
                "text/markdown",
            )
        }

        with self.client.post(
            "/api/v1/ingest",
            files=files,
            name="Ingest Document",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "doc_id" in data:
                    self.doc_ids.append(data["doc_id"])
                    response.success()
                else:
                    response.failure("Missing doc_id in response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(4)
    def get_clauses(self):
        """
        Get clauses for a document.
        Weight: 4 (common after ingestion)
        """
        if not self.doc_ids:
            return

        doc_id = random.choice(self.doc_ids)

        with self.client.get(
            f"/api/v1/clauses?doc_id={doc_id}",
            name="Get Clauses",
            catch_response=True,
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def compile_document(self):
        """
        Compile a document to artifacts.
        Weight: 2 (resource intensive)
        """
        if not self.doc_ids:
            return

        doc_id = random.choice(self.doc_ids)
        formats = random.sample(["yaml", "sql", "json"], k=random.randint(1, 3))

        with self.client.post(
            "/api/v1/compile",
            json={"doc_id": doc_id, "formats": formats},
            name="Compile Document",
            catch_response=True,
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class HeavyUser(HttpUser):
    """
    Simulated heavy user that performs resource-intensive operations.
    """

    wait_time = between(5, 15)
    weight = 1  # Lower weight = fewer of these users

    def on_start(self):
        """Called when a user starts."""
        self.doc_ids = []

    @task(1)
    def ingest_large_document(self):
        """
        Ingest a large policy document.
        """
        # Create a larger document
        sections = []
        for i in range(50):
            sections.append(f"""
## Section {i}

Clause {i}.1: Organizations must comply with requirement {i}.
Clause {i}.2: Employees shall not violate policy {i}.
Clause {i}.3: Customers may request service {i}.
""")

        policy = "# Large Policy Document\n" + "\n".join(sections)

        files = {
            "file": (
                f"large_policy_{random.randint(1000, 9999)}.md",
                policy.encode("utf-8"),
                "text/markdown",
            )
        }

        with self.client.post(
            "/api/v1/ingest",
            files=files,
            name="Ingest Large Document",
            catch_response=True,
            timeout=120,  # Longer timeout for large documents
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "doc_id" in data:
                    self.doc_ids.append(data["doc_id"])
                    response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def compile_all_formats(self):
        """
        Compile a document to all formats.
        """
        if not self.doc_ids:
            return

        doc_id = random.choice(self.doc_ids)

        with self.client.post(
            "/api/v1/compile",
            json={
                "doc_id": doc_id,
                "formats": ["yaml", "sql", "json", "python"],
            },
            name="Compile All Formats",
            catch_response=True,
            timeout=60,
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class ReadOnlyUser(HttpUser):
    """
    Simulated read-only user that only performs queries.
    """

    wait_time = between(0.5, 2)
    weight = 3  # Higher weight = more of these users

    @task(5)
    def health_check(self):
        """Quick health check."""
        self.client.get("/api/v1/health", name="Health Check (ReadOnly)")

    @task(5)
    def list_documents(self):
        """List documents."""
        self.client.get("/api/v1/documents", name="List Documents (ReadOnly)")


# =============================================================================
# Event Hooks
# =============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    if isinstance(environment.runner, MasterRunner):
        print("Running in distributed mode as master")
    print(f"Starting load test against: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("Load test completed")


# =============================================================================
# Custom Shape (Optional)
# =============================================================================

class StagesShape:
    """
    Custom load shape that ramps up in stages.

    Usage:
        locust -f locustfile.py --host http://localhost:8080
        # Then select this shape in the web UI
    """

    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},    # Warm up
        {"duration": 120, "users": 50, "spawn_rate": 5},   # Normal load
        {"duration": 60, "users": 100, "spawn_rate": 10},  # Peak load
        {"duration": 60, "users": 50, "spawn_rate": 5},    # Cool down
        {"duration": 60, "users": 10, "spawn_rate": 2},    # Final stage
    ]

    def tick(self):
        """Return tuple of (users, spawn_rate) or None to stop."""
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])

        return None
