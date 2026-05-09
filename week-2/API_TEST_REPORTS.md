# 📊 FastAPI Load Testing Report (Locust)

## 🧪 Overview

This report summarizes the load testing performed on a FastAPI endpoint:

```
GET /api/v1/customers/customer?customerNumber=103
```

The test was conducted using Locust to simulate concurrent user traffic and evaluate API performance under load.

---

## ⚙️ Test Configuration

* Tool: Locust
* Endpoint: `/api/v1/customers/customer`
* Query Param: `customerNumber=103`
* Simulated Users: 100 (concurrent)
* Test Type: Load / Stress Test
* Duration: ~30 seconds (approx.)

---

## 📈 Performance Summary

| Metric                | Value             |
| --------------------- | ----------------- |
| Total Requests        | 7,268             |
| Failures              | 0 (0%)            |
| Throughput            | ~299 requests/sec |
| Average Response Time | ~331 ms           |
| Median Response Time  | ~200 ms           |
| Max Response Time     | ~9.7 sec          |

---

## 📊 Response Time Distribution

| Percentile | Response Time (ms) |
| ---------- | ------------------ |
| 50%        | 200                |
| 66%        | 210                |
| 75%        | 230                |
| 80%        | 240                |
| 90%        | 370                |
| 95%        | 440                |
| 98%        | 800                |
| 99%        | 9300               |
| 100%       | 9700               |

---

## 📉 Key Observations

### ✅ Strengths

* Zero request failures under load
* Stable average response time (~300 ms)
* Handles ~300 requests/sec effectively
* Most requests served within acceptable latency range

### ⚠️ Issues Identified

* Significant tail latency spikes (99th percentile)
* Some requests taking up to ~9–10 seconds
* Indicates potential bottlenecks under high concurrency

---

## 🔍 Possible Bottlenecks

* Database query latency under load
* Connection pool saturation
* Single-worker or limited worker configuration
* Async task blocking or inefficient query execution

---

## 🚀 Recommendations

### 1. Increase Worker Processes

Run multiple workers:

```bash
uvicorn app.main:app --workers 4
```

---

### 2. Optimize Database Access

* Ensure index on `customerNumber`
* Avoid unnecessary full table scans
* Use connection pooling properly

---

### 3. Monitor Connection Pool

Check for:

* DB connection exhaustion
* Slow queries during peak load

---

### 4. Investigate Tail Latency

Focus on:

* 95th–99th percentile requests
* Identify slow DB queries or blocking operations

---

## 🧠 Conclusion

The API demonstrates strong baseline performance and stability under load with zero failures and ~300 RPS throughput. However, tail latency spikes indicate scalability concerns that should be addressed for production-grade reliability.

---

## 📌 Final Verdict

* **Stability:** ⭐⭐⭐⭐⭐
* **Speed (average):** ⭐⭐⭐⭐
* **Scalability:** ⭐⭐⭐
* **Production readiness:** ⚠️ Needs optimization for high concurrency
