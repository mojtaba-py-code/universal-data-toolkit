# Benchmark results

- Date: 2026-08-08 01:31 UTC
- Rows: 100,000
- Python: 3.12.5
- Platform: Windows-10-10.0.19045-SP0
- CPU: Intel64 Family 6 Model 69 Stepping 1, GenuineIntel

| Scenario | Seconds | Rows/sec | Peak memory (MB) |
| --- | ---: | ---: | ---: |
| row_wise apply | 1.871 | 53,980 | 6.7 |
| vectorised expression | 0.008 | 11,992,543 | 2.3 |
| in_memory pipeline | 1.539 | 65,620 | 24.0 |
| chunked pipeline | 1.449 | 69,722 | 0.0 |
