# Remaining Work

- [ ] Implement rebalancing for heavily skewed buckets (hotspot mitigation).
- [ ] Add metrics collection for rebalancing events (splits, merges).
- [ ] Investigate and fix intermittent failures in `rebalance_survives_split_failures` test (if any persist).
- [ ] Optimize K-Means implementation for faster splitting of very large buckets.
- [ ] Add support for dynamic rebalancing thresholds based on system load.
