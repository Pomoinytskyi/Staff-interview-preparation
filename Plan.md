### Plan:

- **Distributed Systems Foundations**
	- Scaling
	- Latency, throughput, durability
	- Consistency vs availability (CAP)
	- State management fundamentals
	- Logical clocks, time drift, ordering constraints
	- Physical limits of distributed systems

- **Data Storage & Query Models**
	- Relational vs document databases
	- Indexing
	- Sharding
	- Partitioning
	- Database performance principles
	- Storage internals: B-tree, LSM-tree
	- Durability models and trade-offs

- **Consensus & Coordination**
	- Raft, Paxos
	- Leader election
	- Replication modes
	- Vector clocks vs Lamport clocks
	- Distributed transactions

- **Messaging, Events & Streaming**
	- Queues, pub/sub, fan-out
	- Delivery semantics (at-least-once, exactly-once)
	- Indempotency and retry patterns
	- Event sourcing
	- CQRS
	- Sagas (orchestration and choreography)
	- Batch processing
	- Streaming systems

- **Resilience & Fault Tolerance**
	- Circuit breakers
	- Load shedding and graceful degradation
	- Brown-out strategies
	- Backpressure
	- Timeouts and retry control
	- Cascading failure mitigation

- **High Availability & Multi-Region Design**
	- Multi-region patterns
	- Active-active vs active-passive
	- Conflict resolution approaches
	- Global consistency strategies
	- Stateless vs stateful tier design
	- Region failover models

- **Performance Engineering & Scale**
	- Millions-QPS architectural strategies
	- Cache hierarchies (CDN → edge → service → DB)
	- Query optimization
	- Hot path vs cold path design
	- Load testing and benchmarking
	- Concurrency and memory model considerations

- **Observability & Production Excellence**
	- Metrics, logs, tracing
	- SLOs, SLIs, SLAs
	- Alerting philosophy and golden signals
	- Incident response workflows
	- Postmortems and mitigation strategies

- **Deployment & Migration Safety**
	- Zero-downtime migrations
	- Canary and blue-green releases
	- Feature flags
	- Backward and forward compatibility
	- Migration pitfalls
	- Rollback strategies

- **Architecture Strategy & Trade-offs**
	- Scope and boundaries
	- Cross-team dependency mapping
	- Product vs technical prioritization
	- Cost modeling and efficiency
	- Long-term maintainability
	- Decision-making under ambiguity
	- Architecture reviews and RFC processes

- **Leadership & Influence**
	- Leading without authority
	- Ownership boundaries
	- Integrating product and engineering decisions
	- Managing constraints
	- Mentoring senior engineers
	- Structuring teams for sustainable velocity

- **System Design Mastery**
	- End-to-end system design structure
	- Scaling math and capacity reasoning
	- High-scale pattern selection
	- Trade-off and risk analysis
	- Operational considerations
	- Observability baked into design