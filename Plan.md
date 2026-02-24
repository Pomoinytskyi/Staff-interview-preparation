# System Design Component Taxonomy — Deep Dive Topics


## 1. DNS (CoreDNS, BIND, PowerDNS, Unbound, Route 53, Cloud DNS, Azure DNS, Traffic Manager)
1. Recursive vs iterative resolution — full walk through root → TLD → authoritative
2. TTL tuning trade-offs — low TTL for failover speed vs DNS query amplification and latency cost
3. DNS-based global load balancing — latency-based vs geolocation vs weighted vs failover routing policies
4. CNAME vs A vs ALIAS/ANAME — why CNAME can't be used at zone apex and how ALIAS solves it
5. DNS propagation delays — why "propagation" is a misnomer (it's TTL expiry in caches), debugging stale records
6. Zone transfers (AXFR/IXFR) — primary-secondary replication, securing zone transfers with TSIG
7. DNSSEC — chain of trust from root, RRSIG/DNSKEY/DS records, key rotation, why adoption is still low
8. Split-horizon DNS — returning different answers for internal vs external clients, use in hybrid cloud
9. DNS failover and health checking — active-active vs active-passive, health check intervals vs TTL alignment
10. DNS amplification attacks — how open resolvers are exploited for DDoS, response rate limiting (RRL)
11. Negative caching — NXDOMAIN TTL (SOA minimum), impact on failover timing
12. Multi-value answer routing — returning multiple IPs and letting the client choose vs round-robin


## 2. CDN (Cloudflare, Akamai, Fastly, Varnish, CloudFront, Cloud CDN, Azure CDN, Azure Front Door)
1. Cache invalidation strategies — purge vs soft-purge vs versioned URLs (cache busting), invalidation propagation delay across PoPs
2. Origin shielding — collapsing requests to origin through a single mid-tier cache, reducing origin load during cache misses
3. Edge compute — running logic at PoPs (CloudFront Functions, Cloudflare Workers, Fastly Compute), latency vs cold start trade-offs
4. Cache key design — what request attributes to include (path, query params, headers, cookies), over-keying vs under-keying
5. Signed URLs and signed cookies — time-limited access to private content, token generation and validation flow
6. Stale-while-revalidate and stale-if-error — serving expired content while fetching fresh copy, improving availability during origin failures
7. Cache stampede at the edge — thundering herd when TTL expires simultaneously on popular content, request coalescing
8. Dynamic content acceleration — TCP connection reuse to origin, persistent connections, optimized routing between PoPs and origin
9. Multi-tier caching architecture — browser cache → CDN edge → CDN shield → origin cache → origin, how TTLs interact across layers
10. Vary header pitfalls — how Vary on Accept-Encoding, User-Agent, or Cookie can destroy cache hit ratios
11. Range requests and video delivery — byte-range serving for large files, partial cache fills, progressive download vs adaptive bitrate
12. SSL/TLS at edge — certificate management for custom domains, SNI, TLS session resumption across PoPs


## 3. Load Balancing & Reverse Proxy (Nginx, HAProxy, Envoy, Traefik, Caddy, LVS/IPVS)
1. L4 vs L7 balancing — when to use transport-layer (TCP/UDP, high throughput, no inspection) vs application-layer (HTTP, content-based routing)
2. DSR (Direct Server Return) — response bypasses the LB at L4, when and why this matters for throughput-heavy workloads
3. Path-based and host-based routing — routing /api/* to service A and /web/* to service B, regex vs prefix matching at L7
4. Connection multiplexing — many client connections funneled through fewer backend connections, reducing socket exhaustion on backends
5. Sticky sessions — cookie-based affinity at L7, why it fights horizontal scaling, when it's unavoidable (WebSocket, legacy state)
6. SSL/TLS offloading — terminating TLS at LB vs end-to-end encryption (re-encrypt to backend), certificate management at scale
7. Connection pooling to backends — HTTP/2 multiplexing to backend, connection reuse, keep-alive tuning
8. Health check design — TCP vs HTTP health checks, interval/threshold tuning, cascading failure from aggressive health checks, passive health checks
9. Connection draining — gracefully finishing in-flight requests during backend deregistration, timeout tuning
10. Consistent hashing for backend selection — minimizing remapping when backends are added/removed, virtual nodes
11. Cross-zone load balancing — distributing traffic evenly across AZs vs keeping traffic local, cost and latency implications
12. Source IP preservation — PROXY protocol (L4), X-Forwarded-For (L7), how backend sees real client IP through proxy chain
13. Weighted routing for canary deploys — shifting 1% → 5% → 25% → 100% traffic to new version, rollback triggers
14. gRPC and WebSocket load balancing — HTTP/2 stream multiplexing challenges, long-lived connection balancing, connection-level vs request-level
15. Slow loris attacks — how L7 LBs mitigate connection exhaustion, timeout tuning, request body size limits
16. Request/response transformation — adding/removing headers, rewriting paths, injecting correlation IDs
17. HA for the LB itself — VRRP, floating IPs, active-passive LB pairs, ECMP with anycast
18. Hot reload without downtime — Nginx reload vs Envoy hot restart vs HAProxy seamless reload, connection preservation
19. Envoy as universal data plane — xDS API (dynamic config from control plane), outlier detection, circuit breaking, L4+L7 in one proxy


## 4. API Gateway (AWS/Azure API Gateway, Kong, Tyk, KrakenD, APISIX, Apigee)
1. Authentication and authorization at the gateway — JWT validation, OAuth token introspection, API key validation, moving auth out of services
2. Rate limiting algorithms — token bucket vs sliding window vs fixed window, per-consumer vs per-route vs global limits, distributed rate state
3. Request/response transformation — field mapping, protocol translation (REST→gRPC, REST→SOAP), payload modification
4. API versioning strategies — URI path (/v1/), header-based (Accept-Version), query parameter, trade-offs of each
5. Request validation — schema enforcement (JSON Schema, OpenAPI spec) at the gateway, rejecting malformed requests early
6. API composition/aggregation — backend-for-frontend (BFF) pattern, gateway calling multiple services and merging responses, latency impact
7. Canary releases at the gateway — routing percentage of traffic to new API version, header-based routing for testing
8. Throttling vs quota management — short-term burst protection (throttle) vs long-term usage caps (quota), 429 vs 403 semantics
9. Gateway-level caching — caching idempotent GET responses, cache key design, invalidation challenges
10. Analytics and observability — request logging, latency metrics, error rate dashboards, API usage patterns per consumer
11. Cost of gateway as single point of failure — high availability patterns, gateway clustering, what happens when the gateway goes down


## 5. Service Mesh (Istio, Linkerd, Cilium Service Mesh, Consul Connect)
1. Sidecar proxy pattern — how Envoy/linkerd-proxy intercepts all pod traffic, iptables rules for transparent interception
2. mTLS between services — automatic certificate rotation, SPIFFE/SPIRE identity framework, zero-trust networking
3. Traffic splitting — canary deployments via weighted routing at mesh level, mirroring (shadow traffic) for testing
4. Observability injection — automatic distributed tracing header propagation, L7 metrics without code changes
5. Retry and timeout policies — per-route retry budgets, retry storms and cascading failures, retry budget as a percentage of normal traffic
6. Circuit breaking and outlier detection — ejecting unhealthy pods based on consecutive 5xx, recovery probing
7. Fault injection — injecting delays and errors for chaos engineering, testing circuit breakers in production
8. Control plane vs data plane architecture — Istiod as control plane, Envoy sidecars as data plane, configuration propagation delay
9. Performance overhead — latency added per hop (p99 sidecar cost), memory/CPU per sidecar, when the overhead isn't worth it
10. Service mesh without sidecars — eBPF-based mesh (Cilium), ambient mesh (Istio waypoint proxies), trade-offs vs sidecar model


## 6. Service Discovery, Coordination & Configuration (Consul, etcd, ZooKeeper, Eureka, LaunchDarkly, Unleash, Flagsmith)

### Service Discovery
1. Client-side vs server-side discovery — client queries registry directly vs LB resolves on behalf of client, pros/cons of each
2. Self-registration vs third-party registration — service registers itself vs orchestrator registers it, coupling trade-offs
3. DNS-based discovery — SRV records for port discovery, A record round-robin, TTL caching problems causing stale endpoints
4. Consistency model for the registry — CP (Consul, ZooKeeper) vs AP (Eureka) — what happens during network partition
5. Graceful deregistration — draining connections before removing from registry, timing coordination with health checks
6. Multi-datacenter / multi-region discovery — WAN federation (Consul), cross-cluster service resolution, latency-aware routing
7. Kubernetes native discovery — kube-dns/CoreDNS, Service and Endpoints objects, headless services for stateful sets

### Distributed Coordination & Consensus
8. Raft consensus deep dive — leader election, log replication, safety guarantees, what happens during split-brain
9. Paxos vs Raft vs ZAB — trade-offs and practical differences, why Raft won in practice
10. Distributed locking — fencing tokens to prevent stale lock holders from acting, why Redlock is controversial (Martin Kleppmann's analysis)
11. Leader election patterns — using ephemeral nodes (ZooKeeper), session-based leases (etcd), lease renewal and expiry semantics
12. Watch/notification mechanisms — ZooKeeper watches (one-time) vs etcd watches (streaming), ordering guarantees
13. Linearizability vs sequential consistency — what level of consistency does each system actually provide
14. Split-brain scenarios — network partitions causing two leaders, how quorum-based systems prevent this, even-numbered cluster risks
15. Performance under contention — write throughput limits of consensus systems, why they shouldn't be used as general-purpose databases

### Configuration & Feature Flags
16. Feature flags deep dive — boolean flags, multivariate flags, user-targeted flags, percentage rollouts, flag lifecycle (create → enable → clean up)
17. Dark launches — deploying code behind disabled flags, enabling for internal users first, monitoring before public rollout
18. Gradual rollouts — 1% → 10% → 50% → 100%, automatic rollback on error rate spike, sticky bucketing (same user always gets same variant)
19. Configuration propagation — polling vs push vs long-polling, propagation delay, consistency (what if half the fleet has new config)
20. Kill switches — emergency feature disabling without deploy, predefining kill switches for risky features
21. Configuration validation — schema validation before push, canary config deployment, preventing typos from breaking production


## 7. In-Memory Data Store — Cache & Key-Value (Redis, Memcached, DynamoDB, Hazelcast, Dragonfly, RocksDB, FoundationDB, Riak, LevelDB)

### Caching Patterns
1. Cache-aside vs read-through vs write-through vs write-behind vs write-around — when to use each, consistency implications
2. Cache stampede / thundering herd — all requests hit DB when TTL expires on hot key, solutions: locking, probabilistic early expiry, external recomputation
3. Hot key problem — single key receiving disproportionate traffic, solutions: key replication across shards, local in-process caching, key splitting
4. Cache invalidation strategies — event-driven invalidation vs TTL-based, dual-delete pattern for write-through consistency
5. Cache warming strategies — pre-populating after deploy/restart, shadow traffic, lazy loading trade-offs
6. Multi-tier caching — L1 (in-process, Caffeine/Guava) → L2 (distributed, Redis) → DB, consistency between layers

### Redis Deep Dive
7. Eviction policies — LRU vs LFU vs volatile-ttl vs allkeys-random, how Redis implements approximate LRU with sampling
8. Persistence — RDB snapshots vs AOF (append-only file), fsync policies (always, everysec, no), recovery time vs data loss trade-offs
9. Redis Cluster vs Sentinel — cluster mode (16384 hash slots, resharding, cross-slot limitations) vs sentinel (HA for single-shard), when to use which
10. Memory fragmentation — jemalloc behavior, activedefrag, memory overhead per key, how small objects waste memory
11. Data structures beyond strings — sorted sets for leaderboards, HyperLogLog for cardinality, streams for event log, bitmaps for feature flags, geo commands
12. Redis as rate limiter — Lua scripts for atomic token bucket / sliding window, MULTI/EXEC race conditions, distributed rate limiting across instances
13. Redis Pub/Sub — fire-and-forget messaging, no persistence, subscriber must be online, use cases (cache invalidation broadcast, real-time events)
14. Redis Streams — persistent append-only log, consumer groups, acknowledgment, XREADGROUP, comparison with Kafka for lightweight streaming
15. Memcached vs Redis — when Memcached's multi-threaded model wins (pure caching, simple values, no persistence needed)

### Key-Value Store Patterns
16. Partition key design — choosing keys for even distribution, avoiding hot partitions, composite keys for access pattern optimization
17. Consistent hashing internals — virtual nodes, token ring, how DynamoDB/Cassandra distribute data, minimizing remapping during node changes
18. Eventual consistency models — last-writer-wins (LWW), vector clocks, dotted version vectors, conflict resolution strategies
19. Conditional writes and optimistic concurrency — DynamoDB ConditionExpression, compare-and-swap (CAS), preventing lost updates without locks
20. LSM-tree storage engine — write path (memtable → immutable memtable → SSTable), read path (memtable → bloom filter → SSTable), compaction trade-offs
21. DynamoDB single-table design — overloading partition/sort keys, GSI overloading, modeling 1:N and M:N relationships, access pattern-driven design
22. DynamoDB capacity planning — on-demand vs provisioned, adaptive capacity, burst credits, throttling behavior, GSI capacity independence
23. Global tables / multi-region active-active — conflict resolution, replication lag, cross-region consistency trade-offs
24. TTL-based expiry — background deletion process, tombstones, how expired items affect reads before cleanup


## 8. Relational Database (PostgreSQL, MySQL, MariaDB, CockroachDB, TiDB, YugabyteDB, Aurora)
1. Isolation levels deep dive — read uncommitted → read committed → repeatable read → serializable, phantom reads, write skew anomalies, PostgreSQL MVCC implementation
2. B-tree vs LSM-tree index internals — page splits, fill factor, write amplification in LSM, compaction strategies (leveled, tiered, FIFO)
3. Query optimizer internals — cost-based optimization, index selection, join ordering, EXPLAIN ANALYZE reading, common pitfalls (index not used, seq scan)
4. Connection pooling — PgBouncer (transaction vs session vs statement mode), why unpooled connections kill performance at scale
5. Replication topologies — async vs semi-sync vs synchronous, replication lag measurement, promoting replicas, split-brain avoidance
6. Read replicas — eventual consistency challenges, routing reads vs writes, cross-region replicas for DR
7. Sharding strategies — range vs hash vs directory-based, cross-shard queries, resharding pain, application-level vs proxy-level (Vitess, Citus)
8. Write-Ahead Log (WAL) — how WAL enables crash recovery and replication, WAL archiving, point-in-time recovery (PITR)
9. Vacuum and bloat (PostgreSQL) — dead tuple cleanup, autovacuum tuning, table bloat, transaction ID wraparound
10. Deadlock detection and prevention — wait-for graphs, lock ordering, advisory locks, reducing lock contention
11. Partitioning — range, hash, list partitioning, partition pruning, partition-wise joins, how partitioning differs from sharding
12. Schema migration strategies — zero-downtime migrations, expand-and-contract pattern, online DDL (gh-ost, pt-online-schema-change)
13. N+1 query problem — detection, eager loading vs lazy loading, batching strategies, DataLoader pattern
14. Distributed SQL — CockroachDB/TiDB/YugabyteDB architecture, how they achieve ACID across shards, Spanner TrueTime, trade-offs vs single-node RDBMS


## 9. Document Database (MongoDB, CouchDB, RethinkDB, ArangoDB, Firestore, Cosmos DB)
1. Schema design — embedding vs referencing, when to denormalize, bounded vs unbounded arrays, document size limits and growth patterns
2. Indexing strategies — compound indexes and index prefix rule, multikey indexes on arrays, wildcard indexes, partial indexes, index intersection
3. Aggregation pipeline optimization — pipeline ordering ($match early, $project to reduce), allowDiskUse for large datasets, $lookup performance
4. Sharding — range vs hash shard keys, choosing shard key (cardinality, monotonicity, query patterns), jumbo chunks, rebalancing
5. Replica set internals — election protocol, oplog tailing, write concern (w: majority) and read concern (majority, linearizable, snapshot)
6. Change streams — real-time event-driven processing, resume tokens, watch on collection/database/deployment, backpressure handling
7. Transactions in MongoDB — multi-document ACID transactions, performance cost, transaction size limits, when to avoid them
8. Read/write concern tuning — trade-off between durability (w: majority) and latency (w: 1), read preference (primary vs secondaryPreferred)
9. Data modeling anti-patterns — massive arrays, deep nesting, storing large blobs in documents, unnecessary indexing
10. Schema validation — JSON Schema enforcement at DB level, validationAction (warn vs error), migration strategies for schema changes
11. Time-series collections (MongoDB 5.0+) — bucketing, meta fields, automatic expiry, compression advantages over regular collections
12. Cosmos DB specifics — partition key design, RU cost model, consistency levels (strong → eventual), global distribution, multi-model API


## 10. Wide-Column Store (Apache Cassandra, ScyllaDB, HBase, Bigtable)
1. Row key design — composite keys, reverse timestamps for time-series, avoiding hotspots, key length trade-offs
2. Column family design — grouping related columns, storage and compaction implications, wide rows vs tall tables
3. LSM-tree and compaction strategies — size-tiered (STCS) vs leveled (LCS) vs time-window (TWCS), read/write amplification trade-offs
4. Tunable consistency — consistency levels (ONE, QUORUM, LOCAL_QUORUM, ALL), read-repair, anti-entropy, hinted handoff
5. Gossip protocol — peer-to-peer cluster membership, failure detection (phi accrual), seed nodes, topology change propagation
6. Tombstones and deletion — why deletes are expensive, gc_grace_seconds, tombstone accumulation causing read latency spikes
7. Time-series data modeling — TWCS compaction, bucketing by time period, TTL for automatic expiry, partition sizing (target <100MB)
8. Secondary indexes — limitations (scatter-gather queries), materialized views, SAI (Storage-Attached Indexes in Cassandra 4.0)
9. Repair and anti-entropy — full repair vs incremental repair, repair scheduling, what happens when repairs fall behind
10. ScyllaDB vs Cassandra — shard-per-core architecture, seastar framework, when ScyllaDB's C++ implementation wins
11. Bigtable specifics — row key design, column family grouping, tablet splitting, garbage collection policies, Bigtable vs Cassandra trade-offs


## 11. Graph Database (Neo4j, JanusGraph, Dgraph, TigerGraph, ArangoDB, Neptune)
1. Property graph vs RDF model — labeled property graphs (Neo4j) vs triple stores (RDF/SPARQL), when to use which
2. Index-free adjacency — how Neo4j stores direct physical pointers between nodes, O(1) traversal vs index-based lookup
3. Graph traversal optimization — BFS vs DFS trade-offs, traversal depth limits, pruning strategies, avoiding full graph scans
4. Cypher vs Gremlin — declarative (Cypher) vs imperative (Gremlin) query patterns, performance implications
5. Supernodes problem — nodes with millions of edges (celebrities in social graph), query degradation, partitioning strategies
6. Graph partitioning — cutting edges vs replicating vertices, min-cut algorithms, why graph partitioning is fundamentally NP-hard
7. When graph DB beats relational joins — multi-hop traversals, recursive queries, performance cliff in SQL with increasing join depth
8. Graph algorithms at scale — PageRank, community detection, shortest path, connected components, batch frameworks (Pregel, GraphX)


## 12. Time-Series Database (Prometheus, InfluxDB, TimescaleDB, VictoriaMetrics, QuestDB, Kusto/Azure Data Explorer)
1. Compression algorithms — gorilla compression (XOR for timestamps and floats), delta-of-delta encoding, achieving 1-2 bytes per point
2. Downsampling and retention policies — pre-aggregating old data (1s → 1m → 1h), continuous aggregation, tiered storage (hot → warm → cold)
3. High cardinality handling — why high-cardinality labels (user_id) destroy performance, inverted index blowup, solutions (limiting labels, pre-aggregation)
4. Write path optimization — batched writes, out-of-order write handling, WAL for durability, billions of data points per day
5. Time-bucketing strategies — chunking data by time for partition pruning and efficient deletion, chunk sizing trade-offs
6. PromQL and query patterns — rate(), increase(), histogram_quantile(), subquery pitfalls, staleness handling
7. Pull vs push model — Prometheus pull (service discovery + scraping) vs push (InfluxDB/VictoriaMetrics), trade-offs for ephemeral workloads
8. Long-term storage — Thanos/Cortex/VictoriaMetrics for multi-cluster Prometheus, object storage backend, query federation


## 13. Search Engine / Full-Text Search (Elasticsearch, OpenSearch, Solr, Meilisearch, Typesense)
1. Inverted index internals — term dictionary, postings list, skip lists, how a query becomes a set intersection problem
2. Analyzers and tokenization — character filters → tokenizer → token filters pipeline, custom analyzers, language-specific stemming, edge n-grams for autocomplete
3. BM25 scoring — how relevance is calculated, field boosting, function_score for custom ranking (recency, popularity, geolocation)
4. Mapping and schema design — dynamic vs explicit mapping, keyword vs text fields, nested vs object, mapping explosion problem
5. Sharding and replica strategy — shard sizing (target 10-50GB), too many shards problem (cluster state overhead), routing queries to specific shards
6. Near-real-time indexing — refresh interval, how segments are created, force merge for read performance, index lifecycle management (ILM)
7. Aggregations — terms, histogram, date_histogram, composite for pagination, cardinality (HyperLogLog), nested aggregation performance
8. Fuzzy matching and typo tolerance — edit distance (Levenshtein), fuzziness=AUTO, performance cost of fuzzy queries
9. Search relevance tuning — boosting, decay functions, multi_match (best_fields vs cross_fields), A/B testing relevance models
10. Elasticsearch as secondary index — syncing from primary DB, handling consistency lag, zero-downtime reindexing with aliases
11. Elasticsearch for logging — ELK/EFK stack, index-per-day pattern, rollover/ILM, hot-warm-cold architecture, cost vs Loki (label-based, no inverted index)


## 14. Vector Database (Pinecone, Milvus, Weaviate, Qdrant, Chroma, pgvector, FAISS)
1. ANN algorithm internals — HNSW (navigable small world graphs), IVF (inverted file index), PQ (product quantization), recall vs latency vs memory trade-offs
2. Distance metrics — cosine similarity vs euclidean vs dot product, when to use each (normalized vs unnormalized embeddings)
3. Hybrid search — combining vector similarity with keyword/metadata filtering, pre-filter vs post-filter, performance implications
4. Index tuning — HNSW parameters (M, efConstruction, efSearch), IVF nprobe, controlling recall-latency trade-off
5. Quantization for compression — scalar, product, binary quantization, trading precision for memory (16x compression with binary)
6. Embedding dimension trade-offs — 384 vs 768 vs 1536 dimensions, storage cost, latency impact, dimensionality reduction
7. Sharding vectors — partitioning by metadata vs random, keeping semantically similar vectors together vs even distribution
8. Real-time ingestion vs batch indexing — HNSW supports real-time inserts, IVF requires retraining centroids, hybrid approaches
9. RAG architecture patterns — chunking strategies, embedding model selection, retrieval + reranking pipeline, hallucination reduction


## 15. Storage — Object, Block & File (MinIO, Ceph, SeaweedFS, S3 | LVM, ZFS, OpenEBS | NFS, GlusterFS, CephFS, Lustre)

### Object Storage
1. Consistency model — S3 strong read-after-write consistency (since 2020), how this was achieved (witness system)
2. Storage classes and lifecycle — automatic tiering (hot → warm → cold → archive), retrieval latency per class, minimum storage duration charges
3. Multipart upload — parallel upload of large objects, part size tuning, completing/aborting, leftover parts cleanup
4. Presigned URLs — time-limited access without exposing credentials, upload vs download presigned URLs, STS temporary credentials alternative
5. Event notifications — triggering serverless functions on PUT/DELETE, ordering guarantees (or lack thereof), duplicate events
6. Versioning and soft delete — protecting against accidental deletion, MFA delete, lifecycle rules for version cleanup
7. Cross-region replication — async replication, replication lag, conflict resolution (latest timestamp), bidirectional challenges
8. Performance optimization — request parallelism, prefix-based partitioning, byte-range fetches, transfer acceleration
9. Server-side encryption — SSE-S3 vs SSE-KMS vs SSE-C, envelope encryption flow, client-side encryption trade-offs
10. Content-addressable storage — hash-based keys (SHA-256), deduplication, immutability, git-like content storage

### Block Storage
11. IOPS vs throughput vs latency — how they interact, gp3 baseline vs burst vs io2 provisioned
12. Snapshot mechanics — copy-on-write, incremental snapshots, cross-region copy, lazy loading on restore
13. Multi-attach volumes — shared block for active-passive clusters, file system coordination (GFS2, OCFS2)
14. Volume types and selection — SSD (gp3, io2) vs HDD (st1, sc1), matching IOPS needs to workload, cost optimization

### File Storage
15. NFS vs SMB/CIFS — protocol differences, NFS v3 vs v4 (stateful, Kerberos), Linux vs Windows
16. Throughput modes — bursting (credit-based) vs provisioned vs elastic, sizing for workload patterns
17. High-performance parallel file systems — Lustre architecture (MDS + OSS + OST), HPC and ML training workloads
18. POSIX compliance implications — file locking, permission model, how cloud file systems handle POSIX semantics


## 16. Message Queue & Pub/Sub (RabbitMQ, NATS, ActiveMQ, ZeroMQ, BullMQ, SQS, SNS)

### Message Queue Fundamentals
1. Delivery guarantees — at-least-once vs at-most-once vs exactly-once (and why exactly-once is mostly a lie across system boundaries)
2. Visibility timeout tuning — too short = duplicate processing, too long = delayed retry on failure, dynamic extension during processing
3. Dead-letter queues — max receive count configuration, monitoring DLQ depth, replaying DLQ messages, alerting strategies
4. FIFO ordering guarantees — message group IDs for ordered processing per entity, throughput limitations of strict ordering
5. Message deduplication — content-based vs explicit deduplication ID, deduplication window, idempotent consumers as complement
6. Backpressure mechanisms — queue depth monitoring, auto-scaling consumers based on queue metrics, producer slowdown signals
7. Poison pill handling — messages that repeatedly fail processing, isolation strategies, dead-lettering vs manual inspection
8. Priority queues — separate queues per priority vs priority field, starvation prevention for low-priority messages
9. Batch processing — receiving/deleting messages in batches, partial batch failure handling, throughput optimization

### Pub/Sub Patterns
10. Push vs pull delivery — push (webhook/HTTP endpoint) vs pull (consumer polls), at-least-once with acknowledgment, push endpoint reliability
11. Fan-out architectures — SNS + SQS pattern, one event → many independent subscribers, each with own retry/DLQ
12. Message filtering — SNS filter policies (attribute-based), EventBridge content-based rules, filtering at broker vs consumer side
13. Dead-letter topics — messages that can't be delivered after retries, monitoring and replay strategies
14. Message ordering in pub/sub — ordering keys, FIFO topics, trade-offs with throughput

### RabbitMQ Deep Dive
15. Exchange types — direct, fanout, topic, headers exchanges, routing key patterns, binding strategies
16. Quorum queues — Raft-based replication replacing classic mirrored queues, trade-offs and migration path
17. Flow control and memory alarms — backpressure when memory/disk threshold hit, producer blocking, credit-based flow control
18. Prefetch and consumer tuning — prefetch count, manual vs auto acknowledgment, QoS settings

### NATS
19. Core NATS vs JetStream — fire-and-forget core vs persistent JetStream, at-least-once with JetStream, consumer pull vs push
20. Subject-based routing — hierarchical subjects with wildcards (* and >), queue groups for load balancing


## 17. Event Streaming & Schema Management (Apache Kafka, Redpanda, Apache Pulsar, Confluent Schema Registry, Apicurio, Buf)

### Kafka Core
1. Partition design — choosing partition key for even distribution and ordering, partition count planning (can't easily decrease), hot partitions
2. Consumer groups — parallel consumption, partition assignment (range vs round-robin vs sticky), rebalancing storms and cooperative rebalancing
3. Offset management — auto-commit risks, manual commit (at-least-once), exactly-once with transactions, consumer lag monitoring
4. Log compaction — retaining only latest value per key, use cases (changelogs, materialized views), tombstones for deletion
5. Exactly-once semantics (EOS) — idempotent producer, transactional producer, read_committed isolation, performance overhead
6. ISR (In-Sync Replicas) — min.insync.replicas, acks=all behavior, unclean leader election trade-off (availability vs data loss)

### Kafka Producer Tuning
7. Batching and compression — batch.size, linger.ms, compression (snappy vs lz4 vs zstd), buffer.memory and backpressure
8. Acks configuration — acks=0 (fire-and-forget) vs acks=1 (leader only) vs acks=all (ISR), latency vs durability
9. Idempotent producer — enable.idempotence=true, sequence numbers, producer ID, exactly-once within single session

### Kafka Consumer Tuning
10. Poll loop mechanics — max.poll.records, max.poll.interval.ms, session.timeout.ms vs heartbeat.interval.ms, rebalance triggers
11. Consumer lag monitoring — checking lag via consumer group offsets, alerting on growing lag, burrow/kafka-lag-exporter

### Kafka Connect
12. Source and sink connectors — Debezium (CDC source), S3/JDBC sinks, exactly-once delivery, SMT (Single Message Transforms)
13. Distributed vs standalone mode — worker scaling, task rebalancing, connector configuration management

### Kafka Streams
14. KStream vs KTable — event stream vs changelog, stream-table joins, GlobalKTable for broadcast joins
15. Local state stores (RocksDB) — stateful processing, state store backup to changelog topics, interactive queries
16. Windowing in Kafka Streams — tumbling, hopping, sliding, session windows, grace period for late events

### Event Routing
17. Rule-based event routing — EventBridge / Event Grid / Eventarc, content-based filtering, archive and replay, cross-account/cross-region
18. Event bus architecture — central bus vs topic-per-event-type, discovery of event types, coupling trade-offs
19. Dead-letter handling — events that fail delivery after retries, monitoring and replay from DLQ

### Schema Management
20. Avro vs Protobuf vs JSON Schema — Avro (schema in file, compact binary), Protobuf (field numbering, backward compatible by default), JSON Schema (readable but large)
21. Compatibility levels — backward (new reader, old data), forward (old reader, new data), full (both), transitive variants
22. Schema evolution strategies — Avro (add field with default), Protobuf (new field number, never reuse), handling required fields
23. Schema registry in event architecture — producer registers schema, consumer fetches by ID, schema caching, version negotiation
24. Protobuf + gRPC integration — proto file as contract, code generation, backward-compatible service evolution, buf lint/breaking

### Pulsar vs Kafka
25. Pulsar architecture — separate storage (BookKeeper) and serving (brokers), topic-level replication, geo-replication built-in
26. Multi-tenancy — namespace-level isolation, quotas, when Pulsar's architecture wins (many topics, tiered storage)


## 18. Stream Processing (Apache Flink, Spark Structured Streaming, Kafka Streams, Apache Storm, Apache Beam)
1. Windowing semantics — tumbling vs sliding vs session vs global windows, window triggers (event time, processing time, count)
2. Watermarks and late data — how watermarks track event time progress, allowed lateness, side outputs for late events, watermark generation strategies
3. Event time vs processing time — why event time matters (out-of-order events), ingestion time as compromise, skew handling
4. State management — keyed state, operator state, state backends (RocksDB vs heap), state size and checkpointing overhead
5. Checkpointing and exactly-once — Chandy-Lamport algorithm (Flink), checkpoint barriers, aligned vs unaligned checkpoints, interval tuning
6. Backpressure handling — Flink credit-based flow control, Spark micro-batch backpressure, monitoring and resolving bottlenecks
7. Stream-table joins — enriching events with dimension data, temporal joins, slowly changing dimensions (SCD)
8. Savepoints and job versioning — taking savepoints for upgrades, state schema evolution, restoring after code changes
9. Exactly-once end-to-end — idempotent sinks, two-phase commit sinks (Kafka), at-least-once + dedup as alternative
10. Flink vs Spark Streaming vs Kafka Streams — true streaming vs micro-batch vs library, when each architecture wins


## 19. Batch Processing (Apache Spark, Hadoop, Dask, Apache Beam)
1. Spark execution model — driver, executors, stages, tasks, DAG scheduler, how shuffles create stage boundaries
2. Data skew handling — skewed joins (salting, broadcast join), skewed aggregations, monitoring via Spark UI (stage task times)
3. Shuffle optimization — shuffle partitions tuning, avoiding unnecessary shuffles (broadcast join, co-partitioned data), external shuffle service
4. Partitioning strategies — coalesce vs repartition, output partitioning (bucketing, partition columns), partition pruning
5. Fault tolerance — lineage-based recovery (RDD), checkpoint vs recompute trade-offs, speculative execution for stragglers
6. Resource management — YARN vs Kubernetes vs Mesos, dynamic allocation, executor sizing (memory vs cores)
7. Batch vs micro-batch vs streaming — Spark Structured Streaming micro-batch overhead, continuous processing mode, choosing batch interval
8. Data formats — Parquet (columnar, predicate pushdown) vs Avro (row-based, schema evolution) vs ORC, compression (snappy, zstd, gzip)
9. Idempotent batch jobs — handling reruns safely, atomic output writes, partition overwrite vs append, deduplication


## 20. Workflow, Orchestration & Scheduling (Temporal, Apache Airflow, Cadence, Argo Workflows, Prefect, Dagster, Quartz, Celery Beat)

### Workflow Orchestration
1. Saga pattern — orchestration (central coordinator) vs choreography (event-driven), compensating transactions for rollback, partial failure handling
2. Temporal/Cadence deep dive — workflow-as-code, activity retries, signal and query, child workflows, workflow versioning (getVersion/patching)
3. Idempotency in workflows — activity idempotency tokens, handling duplicate workflow starts, deterministic workflow code requirements
4. Long-running workflows — heartbeating for long activities, workflow timeouts, handling workflows that run for days/weeks
5. Workflow versioning — deploying new logic without breaking running instances, Temporal patching, Step Functions aliases
6. State machine design — Step Functions states, choice/parallel/map states, error handling (Catch, Retry), execution history limits
7. Retry policies — exponential backoff with jitter, max retries, non-retryable errors, timeout vs retry interaction

### Pipeline Orchestration (Airflow)
8. Airflow architecture — scheduler, workers, metadata DB, executor types (local, celery, kubernetes), DAG serialization
9. DAG design — dependency management, sensor operators (waiting for data), dynamic DAG generation, XCom for inter-task data passing
10. Backfill and catchup — re-processing historical data, catchup=True/False, ordering guarantees for backfill

### Distributed Scheduling
11. Distributed cron — preventing duplicate execution across instances, leader election for scheduler, distributed locking
12. At-least-once execution — idempotent task design when scheduler fires but task crashes
13. Time zone handling — storing schedules in UTC, DST transitions causing missed or double executions
14. Job dependency chains — DAG-based scheduling, conditional execution, handling upstream failures


## 21. ETL/ELT & Change Data Capture (dbt, Apache NiFi, Airbyte, Fivetran, Debezium, Maxwell, Canal, Meltano)

### ETL vs ELT
1. ETL vs ELT trade-offs — transform before load (ETL: cleaner warehouse) vs load then transform (ELT: leverage warehouse compute)
2. Incremental processing — change detection (timestamps, CDC, watermarks), merge (upsert) strategies, handling late-arriving data
3. Idempotent pipelines — producing same result on re-run, partition overwrite vs merge, deduplication strategies
4. Schema evolution handling — adding/removing columns, type changes, forward/backward compatibility in pipelines
5. Backfill strategies — re-processing historical data without disrupting current pipeline, partition-based backfill
6. Data quality checks — Great Expectations, dbt tests, data freshness monitoring, data contracts between producers and consumers

### dbt Deep Dive
7. dbt models — SQL-based transformations, ref() for dependency management, incremental models (merge strategy, append, delete+insert)
8. Testing framework — unique, not_null, relationships, custom generic tests, documentation generation
9. dbt orchestration — dbt Cloud scheduling, dbt + Airflow, dbt + Dagster, selecting subsets of models with tags/selectors

### Change Data Capture
10. Log-based CDC — tailing database WAL/binlog, no application code changes, minimal overhead on source DB
11. Initial snapshot + streaming — bootstrapping with full table snapshot, switching to incremental, consistency during transition
12. Outbox pattern — writing events to outbox table in same transaction as business data, CDC captures outbox, guarantees exactly-once publish
13. Schema evolution in CDC — handling ALTER TABLE during streaming, Debezium schema history topic, breaking vs compatible changes
14. Ordering guarantees — per-key ordering, partition key in Kafka aligning with primary key, handling multi-table transactions
15. Debezium architecture — Kafka Connect source connector, snapshot mode (initial, schema_only, never), slot management
16. Replication slot management — PostgreSQL WAL retention when slot falls behind, disk space risk, monitoring slot lag
17. Tombstone events — representing DELETE as null-value message, compaction interaction, downstream consumer handling


## 22. Data Warehouse & Data Lake (ClickHouse, Apache Druid, Apache Pinot, Trino, DuckDB, Greenplum | Apache Iceberg, Delta Lake, Apache Hudi, Hive Metastore)

### Data Warehouse
1. Columnar storage internals — column chunks, encoding (dictionary, RLE, bit-packing), predicate pushdown, late materialization
2. Star schema vs snowflake — dimension tables, fact tables, denormalization trade-offs, slowly changing dimensions (SCD type 1/2/3)
3. Separation of storage and compute — BigQuery serverless, Redshift Serverless, scaling compute independently, caching across restarts
4. Partitioning and clustering — partition pruning for time-based queries, clustering for frequently filtered columns, granularity trade-offs
5. Materialized views — pre-computed aggregations, incremental refresh, stale data trade-offs, maintenance cost
6. Query optimization — columnar scan vs index scan, join algorithms (hash, sort-merge, broadcast), execution plan analysis
7. ClickHouse internals — MergeTree engine family, sparse primary index, materialized views as real-time aggregation, ReplacingMergeTree for dedup

### Data Lake
8. Schema-on-read vs schema-on-write — flexibility vs governance, how table formats add schema enforcement back
9. Table formats — Iceberg vs Delta Lake vs Hudi, snapshot isolation, time-travel, schema evolution, partition evolution (Iceberg)
10. File format selection — Parquet (columnar, analytics) vs Avro (row-based, streaming) vs ORC, compression codec selection
11. Small files problem — too many small files killing query performance, compaction strategies, file sizing targets (128MB–1GB)
12. Data catalog and governance — Glue Catalog, Hive Metastore, Unity Catalog, data lineage, classification, access control
13. Partitioning strategy — physical partitioning, partition pruning, over-partitioning vs under-partitioning, Iceberg hidden partitioning
14. ACID transactions on data lakes — how Iceberg/Delta achieve atomicity (metadata-based), concurrent writes, optimistic concurrency

### Lakehouse Pattern
15. Combining lake + warehouse — single system for raw storage and structured queries, eliminating ETL between layers, query engines on open formats


## 23. Serverless / FaaS (OpenFaaS, Knative, Fission, Lambda, Cloud Functions, Azure Functions)
1. Cold start deep dive — JVM vs Node.js vs Python vs Go cold start times, provisioned concurrency, SNAPSTART (Lambda), warming strategies
2. Concurrency models — reserved concurrency (throttling protection), provisioned concurrency (pre-warmed), per-function vs account-level limits
3. Execution model — container reuse between invocations, /tmp persistence, connection pooling challenges (DB connection exhaustion)
4. Event source mapping — SQS, Kinesis, DynamoDB Streams as triggers, batch size and window, partial batch failure handling
5. Cold start mitigation at scale — SnapStart (Java), provisioned concurrency auto-scaling, keep-warm hacks and why they're fragile
6. Cost modeling — per-invocation + duration + memory, break-even point vs always-on containers (typically ~1M req/month)
7. VPC Lambda — ENI attachment delay (historical), Hyperplane improvements, current cold start with VPC
8. Step Functions integration — orchestrating Lambdas, express vs standard workflows, cost at scale


## 24. Container Orchestration & Registry (Kubernetes, Docker Swarm, Nomad | Harbor, Docker Hub, Quay, Nexus)

### Kubernetes Core
1. Pod scheduling deep dive — resource requests vs limits, QoS classes (Guaranteed, Burstable, BestEffort), scheduler predicates and priorities
2. Horizontal Pod Autoscaler — CPU/memory-based scaling, custom metrics (Prometheus adapter), stabilization window, scaling policies
3. Rolling deployment strategies — maxSurge and maxUnavailable, readiness gates, rollback, detecting failed deployments
4. Persistent volumes — PV/PVC lifecycle, storage classes, dynamic provisioning, StatefulSet stable network identity and storage
5. Network policies — pod-to-pod traffic control, ingress/egress rules, CNI plugins (Calico, Cilium), default-deny policies
6. Ingress controllers — Nginx ingress, ALB ingress controller, path-based routing, TLS termination, canary annotations
7. Resource limits and OOMKill — memory limits and OOMKilled pods, CPU throttling (CFS), right-sizing with VPA, resource quotas per namespace
8. Health probes — liveness vs readiness vs startup probes, misconfigured probes causing restart loops, probe timing tuning
9. Node affinity and taints/tolerations — dedicated node pools (GPU, high-memory), pod anti-affinity for HA, topology spread constraints
10. Cluster autoscaler — scaling node pools based on pending pods, scale-down logic (underutilization threshold), node draining
11. RBAC and multi-tenancy — namespace isolation, service accounts, role/clusterrole bindings, pod security standards

### Container Registry
12. Image tagging strategies — avoid :latest in production, semantic versioning, git SHA tags, immutable tags, retention policies
13. Vulnerability scanning — Trivy, Grype, Snyk, scanning in CI vs registry-side, blocking vulnerable images from deploy
14. Image signing and verification — cosign (Sigstore), Notary v2, admission controllers that verify signatures, supply chain trust
15. Multi-architecture images — manifest lists, building for amd64 + arm64 (Graviton), buildx cross-compilation
16. Layer caching and deduplication — how registries deduplicate shared layers, pull optimization, pre-pulling on nodes


## 25. Compute / VMs (KVM, VMware, Proxmox, OpenStack Nova)
1. Instance type selection — compute vs memory vs storage vs GPU optimized, burstable (T-series) credit system, right-sizing methodology
2. Auto-scaling groups — scaling policies (target tracking, step, schedule-based), launch templates, instance refresh for rolling AMI updates
3. Spot/preemptible instances — interruption handling (2-minute warning), spot fleet strategies, mixing on-demand + spot, checkpointing
4. Placement groups — cluster (low latency), spread (max availability), partition (large distributed workloads like HDFS/Cassandra)
5. Machine images — AMI lifecycle, baking images (Packer) vs bootstrapping (user-data), immutable infrastructure pattern
6. Vertical scaling limitations — vCPU and memory ceilings, NUMA architecture, why horizontal scaling is preferred


## 26. Observability — Logging, Metrics, Tracing & Alerting (Prometheus, Grafana, VictoriaMetrics, Thanos, Cortex | Elasticsearch, Loki, Fluentd, Fluent Bit, Vector | Jaeger, Zipkin, OpenTelemetry, Tempo, SigNoz | Alertmanager, PagerDuty, OpsGenie)

### Logging
1. Structured logging design — JSON logs with consistent fields (trace_id, span_id, service, level), avoiding free-text grepping at scale
2. Log aggregation architecture — agent-based (Fluent Bit sidecar) vs daemonset, pipeline (collect → buffer → route → store), backpressure
3. Log sampling strategies — head-based, tail-based (keep logs for errored requests), dynamic sampling based on error rate
4. ELK vs Loki — Elasticsearch (inverted index, full-text) vs Loki (label-based, chunks in object storage), cost and query trade-offs
5. Log retention and cost — hot/warm/cold tiering, log-based metrics (extracting numbers without storing raw logs), when to sample vs store all
6. PII redaction — masking sensitive data in log pipeline, field-level redaction, compliance requirements (GDPR, HIPAA)

### Metrics & Monitoring
7. RED and USE methods — RED (Rate, Errors, Duration) for services, USE (Utilization, Saturation, Errors) for resources
8. SLI/SLO/SLA framework — defining SLIs (latency p99 < 200ms), setting SLO (99.9%), error budget, error budget policies (freeze deploys)
9. Percentiles vs averages — why p99 matters more than mean, coordinated omission problem, histogram vs summary in Prometheus
10. Cardinality management — high-cardinality labels causing metric explosion, relabeling to drop cardinality, label value guidelines
11. Multi-cluster long-term storage — Thanos vs Cortex vs VictoriaMetrics, global query view, object storage backend
12. Alerting on burn rate — multi-window burn rate alerts for SLO-based alerting, avoiding threshold alerts that fire too late
13. Dashboard design — avoiding vanity dashboards, debugging-oriented layouts, USE/RED templates

### Distributed Tracing
14. OpenTelemetry architecture — SDK (instrumentation) → Collector (receive, process, export) → Backend, auto vs manual instrumentation
15. Trace context propagation — W3C TraceContext (traceparent header), B3 format, propagation across async boundaries (queues, cron)
16. Sampling strategies — head-based vs tail-based, probabilistic vs rate-limiting, ParentBased sampling
17. Span modeling — root span, child spans, span kind (client, server, producer, consumer), attributes and events, span status
18. Trace-log-metric correlation — linking trace IDs in logs, exemplars in Prometheus (trace ID on histogram buckets), unified observability
19. Instrumentation of async flows — tracing across Kafka produce/consume, queues, context injection in message headers

### Alerting & Incident Management
20. Alert fatigue reduction — actionable alerts only, grouping, inhibition, silencing, alert ownership
21. Escalation policies — primary → secondary → EM → VP chain, time-based escalation, business hours routing
22. Alertmanager architecture — group_by labels, inhibition (suppress dependent alerts), silencing (maintenance windows), routing tree
23. Runbooks and automation — linking alerts to runbooks, automated remediation, human-in-the-loop for destructive actions
24. Post-mortem process — blameless post-mortems, timeline reconstruction, action items tracking, severity classification


## 27. Authentication & Authorization (Keycloak, Auth0, Ory Hydra/Kratos, Dex, FusionAuth)
1. OAuth 2.0 flows — authorization code + PKCE (SPAs, mobile), client credentials (service-to-service), device code flow, when to use which
2. JWT deep dive — header.payload.signature structure, RS256 vs HS256, token size implications, claims (iss, sub, aud, exp), validation without calling auth server
3. Token lifecycle — short-lived access + refresh tokens, token rotation, revocation challenges with stateless JWT, token introspection endpoint
4. RBAC vs ABAC vs ReBAC — role-based (simple) vs attribute-based (flexible) vs relationship-based (Zanzibar/Google), choosing for scale
5. Session management — server-side sessions (Redis-backed) vs stateless JWT, session fixation, concurrent session limits
6. Service-to-service auth — mTLS, workload identity (K8s service accounts → cloud IAM), SPIFFE, JWT for internal services
7. MFA implementation — TOTP (Google Authenticator), WebAuthn/FIDO2 (passkeys), backup codes, adaptive MFA (risk-based)
8. Google Zanzibar — relationship-based access control at scale, tuple stores, check/expand API, Authzed/SpiceDB implementation


## 28. Secrets Management & Encryption (HashiCorp Vault, SOPS, Let's Encrypt, cert-manager, OpenSSL, step-ca)

### Secrets
1. Envelope encryption — data key encrypts data, master key encrypts data key, why two layers (key rotation, HSM constraints)
2. Dynamic secrets — generating short-lived DB credentials on-demand (Vault), automatic revocation, blast radius reduction
3. Secret rotation — zero-downtime rotation (dual-secret strategy), Lambda-based rotation, rotation interval trade-offs
4. Vault architecture — seal/unseal, Shamir's Secret Sharing, auto-unseal with cloud KMS, storage backends, HA mode
5. Secret injection patterns — env vars (visible in process table) vs mounted files vs init container vs CSI Secrets Store driver
6. Audit logging — every secret access logged, anomalous access detection, compliance requirements

### Encryption & PKI
7. TLS handshake deep dive — ClientHello → ServerHello → certificate → key exchange, TLS 1.3 (1-RTT), 0-RTT resumption risks
8. mTLS — both parties present certificates, validation chain, use in service mesh, bootstrapping trust
9. Certificate lifecycle — issuance, renewal, revocation (CRL vs OCSP vs OCSP stapling), automated renewal (cert-manager, ACM)
10. HSM vs software KMS — FIPS 140-2 levels, when HSM is required (compliance, key ceremony), latency implications
11. Key rotation — rotating KEK without re-encrypting all data (envelope encryption benefit), rotating DEK, key versioning
12. Client-side encryption — encrypting before sending to storage, key management complexity, when required (zero-trust, regulatory)


## 29. Edge Security — WAF & DDoS (Cloudflare WAF, ModSecurity, Coraza, NAXSI)
1. OWASP Core Rule Set — SQLi, XSS, RCE, LFI/RFI detection rules, false positive tuning (paranoia levels), rule exclusion patterns
2. Rate limiting rules — per-IP, per-URI, per-session rate rules, distributed rate limiting across edge nodes
3. Bot mitigation — distinguishing good bots (Googlebot) from bad, CAPTCHA challenges, JavaScript challenges, behavioral analysis
4. Custom rules — pattern matching on headers/body/URI, IP reputation lists, geo-blocking
5. False positive management — logging-only mode before blocking, tuning rules for specific paths, whitelisting trusted sources
6. WAF placement — at CDN edge vs at load balancer, latency and coverage trade-offs
7. DDoS attack types — volumetric (UDP flood, DNS amplification), protocol (SYN flood), application-layer (HTTP flood, slow POST)
8. Anycast-based mitigation — distributing attack traffic across global PoPs, absorbing at the edge
9. Traffic scrubbing — rerouting through scrubbing centers, BGP-based rerouting, clean traffic forwarded to origin
10. Application-layer DDoS — L7 attacks that look like legitimate traffic, behavioral fingerprinting, challenge-response
11. Always-on vs on-demand — always-on (Cloudflare) vs on-demand scrubbing, detection-to-mitigation time


## 30. Resilience Patterns — Rate Limiting & Circuit Breaking (resilience4j, Polly, Hystrix)

### Rate Limiting
1. Token bucket algorithm — bucket capacity (burst), refill rate (sustained), implementation with timestamps
2. Sliding window log — storing each request timestamp, precise but memory-expensive, Redis sorted set implementation
3. Sliding window counter — hybrid of fixed window and sliding log, weighted count, good accuracy with low memory
4. Fixed window problems — boundary burst (2x rate at window boundary), simple but imprecise
5. Distributed rate limiting — centralized counter in Redis, race conditions (use Lua scripts), eventual consistency trade-off
6. Client identification — rate limit by API key, user ID, IP, or combination, handling proxies and shared IPs
7. Hierarchical rate limits — global → per-service → per-user → per-endpoint, which layer enforces what
8. Graceful handling — 429 response with Retry-After header, X-RateLimit-Remaining/Reset headers, client-side backoff

### Circuit Breaking
9. State machine — closed (normal) → open (fail-fast) → half-open (probe), transition thresholds and timers
10. Failure detection — error count vs error percentage, slow call detection (latency threshold), which errors count (5xx yes, 4xx no)
11. Half-open state — limited probe requests, success threshold to close, failure in half-open immediately re-opens
12. Fallback strategies — cached response, default value, degraded functionality, queue-for-later, composing fallbacks in call chains
13. Bulkhead pattern — isolating thread/connection pools per dependency, preventing one slow service from exhausting all resources
14. Retry + circuit breaker interaction — retries with exponential backoff + jitter, retry budget limiting total traffic, retries triggering circuit breaker
15. Circuit breaker in distributed systems — per-instance vs shared state, cascading opening across service graph
16. Monitoring — metrics for open/closed/half-open transitions, alerts when circuits open frequently


## 31. CI/CD & Infrastructure as Code (GitHub Actions, GitLab CI, Jenkins, Argo CD, Tekton, CircleCI | Terraform, OpenTofu, Pulumi, Ansible, Crossplane)

### CI/CD
1. Deployment strategies — rolling (incremental replace), blue-green (instant switch), canary (gradual shift), shadow/dark deployment
2. Rollback mechanisms — automatic rollback on health check failure, database rollback challenges, forward-fix vs rollback culture
3. Build caching — layer caching (Docker), dependency caching, remote build caches, cache invalidation
4. GitOps — Argo CD / Flux CD, declarative desired state in git, pull-based deployment, drift reconciliation
5. Trunk-based development — short-lived feature branches, feature flags for incomplete features, release branching anti-patterns
6. Pipeline security — SAST/DAST/SCA scanning, secret scanning, signed artifacts, supply chain security (SLSA, Sigstore)
7. Environment promotion — dev → staging → prod pipeline, environment parity, promotion gates (manual approval, automated tests)
8. Artifact management — immutable versioning, container image tagging, artifact promotion vs rebuild

### Infrastructure as Code
9. Declarative vs imperative — Terraform (desired state) vs Pulumi/CDK (programming language), trade-offs
10. State management — Terraform state file, remote backends (S3 + DynamoDB lock), state locking, corruption recovery
11. Drift detection — detecting manual changes (terraform plan), reconciliation strategies, preventing drift (RBAC, CI-only deploys)
12. Module design — reusable modules, input/output contracts, versioning, module registry
13. Secret handling in IaC — avoiding secrets in state file, referencing Vault/Secrets Manager, encrypted state backends
14. Testing IaC — unit tests (Terratest), plan validation, policy-as-code (OPA, Sentinel, Checkov), preview environments
15. Import existing resources — terraform import, gradual IaC adoption strategy


## 32. Real-Time Communication (Socket.IO, Centrifugo, Mercure, Ably, Pusher)
1. WebSocket vs SSE vs long polling — bidirectional (WebSocket) vs server-push (SSE), reconnection semantics, HTTP/2 compatibility
2. Horizontal scaling WebSockets — sticky sessions vs pub/sub fan-out (Redis, Kafka), connection state in stateless infrastructure
3. Connection management at scale — heartbeat/ping-pong, idle cleanup, memory per connection, 100K+ concurrent connections per node
4. Presence detection — tracking online users, distributed presence with Redis pub/sub, heartbeat-based timeout, millions of users
5. Message ordering — per-channel ordering, global ordering challenges, vector clocks for conflict resolution
6. Reconnection and offline sync — client-side queue during disconnect, last-event-ID (SSE), cursor-based catch-up on reconnect
7. Fan-out architecture — one publish → millions of subscribers (live sports, stock tickers), tiered fan-out, edge fan-out


## 33. Notification & Email (Novu, ntfy, SendGrid, Twilio, Mailgun, Postfix, Postal)

### Notification Service
1. Multi-channel orchestration — deciding channel (push, email, SMS, in-app) based on urgency and user preference, fallback chains
2. Delivery guarantees per channel — push delivery is best-effort (device offline), email bounce handling, SMS delivery receipts
3. User preference management — opt-in/opt-out per channel, quiet hours, frequency capping (max N per hour)
4. Push notification architecture — APNS and FCM flows, device token management, token invalidation, silent push for background updates
5. Template management — localization, personalization variables, A/B testing content, rich notifications (images, actions)

### Email Service
6. Email authentication — SPF (authorized senders), DKIM (cryptographic signature), DMARC (policy enforcement), alignment requirements
7. Bounce handling — hard bounce (invalid address) vs soft bounce (mailbox full), feedback loops (complaints), list hygiene
8. Sending reputation — dedicated vs shared IP, warm-up schedule, domain reputation, sender score monitoring
9. Deliverability optimization — authentication, engagement tracking, proper unsubscribe headers (List-Unsubscribe)
10. Transactional vs marketing — separate streams (different IPs/domains), CAN-SPAM/GDPR compliance
11. Email at scale — MTA architecture, queue management, retry scheduling, connection pooling to receiving servers


## 34. Media Processing & Content Management (FFmpeg, Sharp, Imgproxy, Thumbor, HandBrake, Tus protocol)

### Media Processing
1. Adaptive bitrate streaming — HLS vs DASH, manifest files (m3u8/mpd), segment duration trade-offs, multi-rendition encoding
2. Codec selection — H.264 (broad compat) vs H.265 (50% smaller) vs VP9 vs AV1 (best compression, slow encode), browser support
3. Transcoding pipeline architecture — job queue, parallel encoding for multiple renditions, priority queues for live vs VOD
4. Image optimization — format selection (WebP, AVIF, JPEG fallback), quality-size curve, responsive images (srcset)
5. DRM — Widevine, FairPlay, PlayReady, license server architecture, key rotation, offline playback
6. Live streaming — ingest (RTMP/SRT) → transcoding → packaging → CDN → player, latency tiers (standard 20-30s, low <5s, ultra-low <1s)

### Content / Blob Processing
7. Resumable upload — Tus protocol, multipart upload with checkpointing, handling unreliable mobile connections
8. Processing pipeline design — event-driven (S3 event → queue → processor), fan-out for multiple output formats
9. Virus scanning — ClamAV integration, quarantine bucket pattern, scanning before serving
10. Content-addressable storage — hash-based keys for deduplication, immutability guarantees
11. Thumbnail and preview generation — on-upload (eager) vs on-first-request (lazy), caching generated assets
12. Content moderation — AI moderation (Rekognition, Cloud Vision), pre-publish review queue, appeal workflow


## 35. Geospatial / Location (PostGIS, OpenStreetMap, OSRM, H3, Nominatim, Mapbox)
1. Geohashing — encoding lat/lng into hierarchical string, prefix-based proximity search, edge cases at geohash boundaries
2. Spatial indexing — R-tree (range queries), quadtree (recursive subdivision), K-D tree, PostGIS GIST indexes
3. Proximity search at scale — geohash prefix scan, bounding box + Haversine refinement, H3 hexagonal grid (uniform area cells)
4. Geofencing — point-in-polygon algorithms, real-time geofence triggering, scaling to millions of geofences
5. Route optimization — Dijkstra vs A* vs contraction hierarchies, real-time traffic, multi-stop optimization (TSP approximation)
6. PostGIS — geometry vs geography types, ST_DWithin for proximity, spatial joins, index-only scans on spatial data


## 36. ID Generation (Snowflake, ULID, UUID v4/v7, KSUID, NanoID, Sonyflake)
1. UUID v4 vs v7 — randomness vs sortability, B-tree index fragmentation from random UUIDs, why v7 matters for DB performance
2. Snowflake ID design — 41-bit timestamp + 10-bit machine ID + 12-bit sequence, clock skew handling, machine ID assignment
3. ULID — lexicographically sortable, Crockford's Base32, monotonic sort within same millisecond, UUID storage compatibility
4. Collision probability — birthday problem math, why 128-bit randomness is sufficient for most systems
5. ID as partition key — sortable IDs cause hot partition on latest partition, random IDs lose time-ordering, compromise strategies
6. Database auto-increment at scale — single point of failure, multi-master conflicts, block allocation (Flickr ticket server)
7. K-sortable IDs — KSUID (128-bit, seconds resolution), benefits for indexing, time-extraction from ID, privacy implications