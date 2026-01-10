#!/usr/bin/env python3
"""
AegisLang Stress Testing Script.

This script performs stress testing on the AegisLang API without
requiring external dependencies like locust.

Usage:
    python tests/performance/stress_test.py --url http://localhost:8080
    python tests/performance/stress_test.py --url http://localhost:8080 --users 50 --duration 120
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import urllib.error
import urllib.parse


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TestConfig:
    """Test configuration."""
    base_url: str = "http://localhost:8080"
    num_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    think_time_min: float = 0.5
    think_time_max: float = 2.0


@dataclass
class RequestResult:
    """Result of a single request."""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TestResults:
    """Aggregated test results."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    requests_by_endpoint: Dict[str, List[RequestResult]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_POLICIES = [
    """# AML Policy

Financial institutions must verify customer identity.
Banks shall report suspicious transactions.
Employees shall not share confidential data.
""",
    """# Data Policy

Organizations must obtain consent for data collection.
Users may request data deletion.
Staff shall protect personal information.
""",
    """# HR Policy

Employees must complete required training.
Managers shall conduct performance reviews.
Staff shall not discriminate.
""",
]


# =============================================================================
# HTTP Client
# =============================================================================

def make_request(
    url: str,
    method: str = "GET",
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30,
) -> tuple:
    """Make an HTTP request and return (status_code, response_body, elapsed_time)."""
    start = time.time()

    if headers is None:
        headers = {}

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            elapsed = time.time() - start
            return response.status, body, elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        return e.code, str(e), elapsed
    except urllib.error.URLError as e:
        elapsed = time.time() - start
        return 0, str(e.reason), elapsed
    except Exception as e:
        elapsed = time.time() - start
        return 0, str(e), elapsed


# =============================================================================
# Test Scenarios
# =============================================================================

def test_health_check(config: TestConfig) -> RequestResult:
    """Test health check endpoint."""
    url = f"{config.base_url}/api/v1/health"
    status, _, elapsed = make_request(url)

    return RequestResult(
        endpoint="/api/v1/health",
        method="GET",
        status_code=status,
        response_time=elapsed,
        success=status == 200,
        error=None if status == 200 else f"Status: {status}",
    )


def test_list_documents(config: TestConfig) -> RequestResult:
    """Test list documents endpoint."""
    url = f"{config.base_url}/api/v1/documents"
    status, _, elapsed = make_request(url)

    return RequestResult(
        endpoint="/api/v1/documents",
        method="GET",
        status_code=status,
        response_time=elapsed,
        success=status == 200,
        error=None if status == 200 else f"Status: {status}",
    )


def test_ingest_document(config: TestConfig) -> RequestResult:
    """Test document ingestion."""
    url = f"{config.base_url}/api/v1/ingest"
    policy = random.choice(SAMPLE_POLICIES)

    # Create multipart form data manually
    boundary = f"----WebKitFormBoundary{random.randint(100000, 999999)}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="policy_{random.randint(1000, 9999)}.md"\r\n'
        f"Content-Type: text/markdown\r\n\r\n"
        f"{policy}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }

    status, response, elapsed = make_request(
        url, method="POST", data=body, headers=headers
    )

    success = status == 200
    doc_id = None
    if success:
        try:
            data = json.loads(response)
            doc_id = data.get("doc_id")
        except json.JSONDecodeError:
            pass

    return RequestResult(
        endpoint="/api/v1/ingest",
        method="POST",
        status_code=status,
        response_time=elapsed,
        success=success,
        error=None if success else f"Status: {status}",
    )


# =============================================================================
# Virtual User
# =============================================================================

class VirtualUser:
    """Simulated user that makes requests."""

    def __init__(self, user_id: int, config: TestConfig, results: TestResults):
        self.user_id = user_id
        self.config = config
        self.results = results
        self.running = False

    def run(self, duration: float):
        """Run user simulation for specified duration."""
        self.running = True
        end_time = time.time() + duration

        # Weighted task selection
        tasks = [
            (test_health_check, 10),
            (test_list_documents, 5),
            (test_ingest_document, 2),
        ]

        total_weight = sum(w for _, w in tasks)

        while self.running and time.time() < end_time:
            # Select task based on weight
            r = random.uniform(0, total_weight)
            cumulative = 0
            selected_task = tasks[0][0]

            for task, weight in tasks:
                cumulative += weight
                if r <= cumulative:
                    selected_task = task
                    break

            # Execute task
            try:
                result = selected_task(self.config)
                self._record_result(result)
            except Exception as e:
                self._record_error(str(e))

            # Think time
            think_time = random.uniform(
                self.config.think_time_min, self.config.think_time_max
            )
            time.sleep(think_time)

    def stop(self):
        """Stop the user simulation."""
        self.running = False

    def _record_result(self, result: RequestResult):
        """Record a request result."""
        self.results.total_requests += 1

        if result.success:
            self.results.successful_requests += 1
        else:
            self.results.failed_requests += 1
            if result.error:
                self.results.errors.append(result.error)

        self.results.response_times.append(result.response_time)

        if result.endpoint not in self.results.requests_by_endpoint:
            self.results.requests_by_endpoint[result.endpoint] = []
        self.results.requests_by_endpoint[result.endpoint].append(result)

    def _record_error(self, error: str):
        """Record an error."""
        self.results.failed_requests += 1
        self.results.errors.append(error)


# =============================================================================
# Test Runner
# =============================================================================

def run_stress_test(config: TestConfig) -> TestResults:
    """Run the stress test."""
    results = TestResults()
    results.start_time = time.time()

    print(f"\n{'='*60}")
    print("AegisLang Stress Test")
    print(f"{'='*60}")
    print(f"Target: {config.base_url}")
    print(f"Users: {config.num_users}")
    print(f"Duration: {config.duration_seconds}s")
    print(f"Ramp-up: {config.ramp_up_seconds}s")
    print(f"{'='*60}\n")

    # Check if server is available
    print("Checking server availability...")
    try:
        status, _, _ = make_request(f"{config.base_url}/api/v1/health", timeout=5)
        if status != 200:
            print(f"Warning: Health check returned status {status}")
    except Exception as e:
        print(f"Error: Server not available - {e}")
        return results

    print("Server is available. Starting test...\n")

    # Create thread pool
    users = []
    with ThreadPoolExecutor(max_workers=config.num_users) as executor:
        # Ramp up users gradually
        ramp_interval = config.ramp_up_seconds / max(config.num_users, 1)

        futures = []
        for i in range(config.num_users):
            user = VirtualUser(i, config, results)
            users.append(user)

            # Calculate duration for this user
            delay = i * ramp_interval
            user_duration = config.duration_seconds - delay

            if user_duration > 0:
                future = executor.submit(user.run, user_duration)
                futures.append(future)

            time.sleep(ramp_interval)

        # Wait for all users to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"User error: {e}")

    results.end_time = time.time()
    return results


def print_results(results: TestResults):
    """Print test results summary."""
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}\n")

    duration = (results.end_time or time.time()) - (results.start_time or 0)

    print(f"Duration: {duration:.2f}s")
    print(f"Total Requests: {results.total_requests}")
    print(f"Successful: {results.successful_requests}")
    print(f"Failed: {results.failed_requests}")

    if results.total_requests > 0:
        success_rate = (results.successful_requests / results.total_requests) * 100
        print(f"Success Rate: {success_rate:.1f}%")

    if duration > 0:
        rps = results.total_requests / duration
        print(f"Requests/sec: {rps:.2f}")

    if results.response_times:
        print(f"\nResponse Times:")
        print(f"  Min: {min(results.response_times)*1000:.2f}ms")
        print(f"  Max: {max(results.response_times)*1000:.2f}ms")
        print(f"  Avg: {statistics.mean(results.response_times)*1000:.2f}ms")
        print(f"  Median: {statistics.median(results.response_times)*1000:.2f}ms")

        if len(results.response_times) > 1:
            print(f"  Std Dev: {statistics.stdev(results.response_times)*1000:.2f}ms")

        # Percentiles
        sorted_times = sorted(results.response_times)
        p90_idx = int(len(sorted_times) * 0.90)
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)

        print(f"  P90: {sorted_times[p90_idx]*1000:.2f}ms")
        print(f"  P95: {sorted_times[p95_idx]*1000:.2f}ms")
        print(f"  P99: {sorted_times[p99_idx]*1000:.2f}ms")

    print(f"\nResults by Endpoint:")
    for endpoint, requests in results.requests_by_endpoint.items():
        success_count = sum(1 for r in requests if r.success)
        times = [r.response_time for r in requests]
        avg_time = statistics.mean(times) if times else 0

        print(f"  {endpoint}:")
        print(f"    Count: {len(requests)}")
        print(f"    Success: {success_count}/{len(requests)}")
        print(f"    Avg Time: {avg_time*1000:.2f}ms")

    if results.errors:
        unique_errors = set(results.errors[:10])  # First 10 unique errors
        print(f"\nErrors (first 10 unique):")
        for error in unique_errors:
            print(f"  - {error}")

    print(f"\n{'='*60}\n")


def export_results(results: TestResults, filename: str):
    """Export results to JSON file."""
    data = {
        "summary": {
            "start_time": results.start_time,
            "end_time": results.end_time,
            "duration": (results.end_time or 0) - (results.start_time or 0),
            "total_requests": results.total_requests,
            "successful_requests": results.successful_requests,
            "failed_requests": results.failed_requests,
            "success_rate": (
                results.successful_requests / results.total_requests * 100
                if results.total_requests > 0
                else 0
            ),
        },
        "response_times": {
            "min": min(results.response_times) if results.response_times else 0,
            "max": max(results.response_times) if results.response_times else 0,
            "avg": statistics.mean(results.response_times) if results.response_times else 0,
            "median": statistics.median(results.response_times) if results.response_times else 0,
        },
        "by_endpoint": {},
    }

    for endpoint, requests in results.requests_by_endpoint.items():
        times = [r.response_time for r in requests]
        data["by_endpoint"][endpoint] = {
            "count": len(requests),
            "success": sum(1 for r in requests if r.success),
            "failed": sum(1 for r in requests if not r.success),
            "avg_time": statistics.mean(times) if times else 0,
        }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results exported to: {filename}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AegisLang Stress Testing Script"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the API (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Number of concurrent users (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--ramp-up",
        type=int,
        default=10,
        help="Ramp-up time in seconds (default: 10)",
    )
    parser.add_argument(
        "--output",
        help="Export results to JSON file",
    )

    args = parser.parse_args()

    config = TestConfig(
        base_url=args.url,
        num_users=args.users,
        duration_seconds=args.duration,
        ramp_up_seconds=args.ramp_up,
    )

    results = run_stress_test(config)
    print_results(results)

    if args.output:
        export_results(results, args.output)

    # Exit with error code if too many failures
    if results.total_requests > 0:
        failure_rate = results.failed_requests / results.total_requests
        if failure_rate > 0.1:  # More than 10% failure
            print("Warning: High failure rate detected")
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
