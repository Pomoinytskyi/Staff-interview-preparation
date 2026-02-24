# System Design Component Taxonomy — Deep Dive Questions



## 1. DNS (Route 53, Cloud DNS, Azure DNS, Traffic Manager, BIND, CoreDNS, PowerDNS, Unbound)
1. Recursive vs iterative resolution — full walk through root → TLD → authoritative,
2. TTL tuning trade-offs — low TTL for failover speed vs DNS query amplification and latency cost,
3. DNS-based global load balancing — latency-based vs geolocation vs weighted vs failover routing policies,
4. CNAME vs A vs ALIAS/ANAME — why CNAME can't be used at zone apex and how ALIAS solves it,
5. DNS propagation delays — why "propagation" is a misnomer (it's TTL expiry in caches), debugging stale records,
6. Zone transfers (AXFR/IXFR) — primary-secondary replication, securing zone transfers with TSIG,
7. DNSSEC — chain of trust from root, RRSIG/DNSKEY/DS records, key rotation, why adoption is still low,
8. Split-horizon DNS — returning different answers for internal vs external clients, use in hybrid cloud,
9. DNS failover and health checking — active-active vs active-passive, health check intervals vs TTL alignment,
10. DNS amplification attacks — how open resolvers are exploited for DDoS, response rate limiting (RRL),
11. Negative caching — NXDOMAIN TTL (SOA minimum), impact on failover timing,
12. Multi-value answer routing — returning multiple IPs and letting the client choose vs round-robin



## 2. CDN (CloudFront, Cloud CDN, Azure CDN, Azure Front Door, Cloudflare, Akamai, Fastly, Varnish, Nginx with caching)
1.  Cache invalidation strategies — purge vs soft-purge vs versioned URLs (cache busting), invalidation propagation delay across PoPs,
2.  Origin shielding — collapsing requests to origin through a single mid-tier cache, reducing origin load during cache misses,
3.  Edge compute — running logic at PoPs (CloudFront Functions, Cloudflare Workers, Fastly Compute), latency vs cold start trade-offs,
4.  Cache key design — what request attributes to include (path, query params, headers, cookies), over-keying vs under-keying,
5.  Signed URLs and signed cookies — time-limited access to private content, token generation and validation flow,
6.  Stale-while-revalidate and stale-if-error — serving expired content while fetching fresh copy, improving availability during origin failures,
7.  Cache stampede at the edge — thundering herd when TTL expires simultaneously on popular content, request coalescing,
8.  Dynamic content acceleration — TCP connection reuse to origin, persistent connections, optimized routing between PoPs and origin,
9.  Multi-tier caching architecture — browser cache → CDN edge → CDN shield → origin cache → origin, how TTLs interact across layers,
10. Vary header pitfalls — how Vary on Accept-Encoding, User-Agent, or Cookie can destroy cache hit ratios,
11. Range requests and video delivery — byte-range serving for large files, partial cache fills, progressive download vs adaptive bitrate,
12. SSL/TLS at edge — certificate management for custom domains, SNI, TLS session resumption across PoPs



## 3a. L4 Load Balancer (NLB, GCP Network LB, Azure Load Balancer, HAProxy TCP, LVS, IPVS, Envoy L4)
1. DSR (Direct Server Return) — response bypasses the LB, when and why this matters for throughput,
2. Connection draining — gracefully finishing in-flight requests during backend deregistration, timeout tuning,
3. Consistent hashing for backend selection — minimizing remapping when backends are added/removed,
4. Health check design — TCP vs HTTP health checks, interval/threshold tuning, cascading failure from aggressive health checks,
5. Cross-zone load balancing — distributing traffic evenly across AZs vs keeping traffic local, cost implications,
6. Source IP preservation — PROXY protocol, X-Forwarded-For unavailable at L4, how backend sees real client IP,
7. Flow-based vs packet-based balancing — how stateful tracking (connection tables) works, memory/scale limits,
8. TCP connection idle timeout — mismatch between LB timeout and backend/client keepalive causing silent drops,
9. High availability for the LB itself — VRRP, floating IPs, active-passive LB pairs, ECMP with anycast



## 3b. L7 Load Balancer (ALB, GCP HTTP(S) LB, Azure Application Gateway, Azure Front Door, Nginx, HAProxy HTTP, Envoy, Traefik, Caddy)
1. Path-based and host-based routing — routing /api/* to service A and /web/* to service B, regex vs prefix matching,
2. Sticky sessions — cookie-based affinity, why it fights horizontal scaling, when it's unavoidable (WebSocket, legacy state),
3. SSL/TLS offloading — terminating TLS at LB vs end-to-end encryption (re-encrypt to backend), certificate management,
4. Connection pooling and multiplexing — reusing backend connections across many client connections, HTTP/2 to backend,
5. Weighted routing for canary deploys — shifting 1% → 5% → 25% → 100% traffic to new version, rollback triggers,
6. Request and response transformation — adding/removing headers, rewriting paths, injecting correlation IDs,
7. Graceful degradation — serving cached responses or static fallbacks when all backends are unhealthy,
8. Rate limiting at the LB layer — per-client, per-path, per-header rate limits vs application-level rate limiting,
9. Slow loris and slowloris-style attacks — how L7 LBs mitigate connection exhaustion attacks, timeout tuning,
10. gRPC and WebSocket load balancing — HTTP/2 multiplexing challenges, long-lived connection balancing



## 4. Reverse Proxy (Nginx, HAProxy, Envoy, Traefik, Caddy, Apache HTTP Server)
1. Connection multiplexing — many client connections funneled through fewer backend connections, reducing socket exhaustion,
2. Request buffering and spooling — absorbing slow clients to free backend threads, proxy_buffering implications,
3. Backend health checks — active (periodic probes) vs passive (monitoring response codes), removal and re-addition logic,
4. Upstream hashing — routing requests for the same resource to the same backend (cache affinity), consistent hashing,
5. Proxy protocol — preserving original client IP/port across L4 proxies, PROXY protocol v1 vs v2,
6. Hot reload without downtime — Nginx reload vs Envoy hot restart vs HAProxy seamless reload, connection preservation,
7. Compression and decompression — offloading gzip/brotli from backends, CPU cost vs bandwidth savings,
8. mTLS termination — proxy verifying client certificates, forwarding client identity to backend via headers,
9. Circuit breaking at proxy level — Envoy outlier detection, ejecting unhealthy upstreams automatically



## 5. API Gateway (API Gateway REST/HTTP, AppSync, Apigee, Cloud Endpoints, GCP API Gateway, Azure API Management, Kong, Tyk, KrakenD, APISIX)
1. Authentication and authorization at the gateway — JWT validation, OAuth token introspection, API key validation, moving auth out of services,
2. Rate limiting algorithms — token bucket vs sliding window vs fixed window, per-consumer vs per-route vs global limits, distributed rate state,
3. Request/response transformation — field mapping, protocol translation (REST→gRPC, REST→SOAP), payload modification,
4. API versioning strategies — URI path (/v1/), header-based (Accept-Version), query parameter, trade-offs of each,
5. Request validation — schema enforcement (JSON Schema, OpenAPI spec) at the gateway, rejecting malformed requests early,
6. API composition/aggregation — backend-for-frontend (BFF) pattern, gateway calling multiple services and merging responses, latency impact,
7. Canary releases at the gateway — routing percentage of traffic to new API version, header-based routing for testing,
8. Throttling vs quota management — short-term burst protection (throttle) vs long-term usage caps (quota), 429 vs 403 semantics,
9. Gateway-level caching — caching idempotent GET responses, cache key design, invalidation challenges,
10. Analytics and observability — request logging, latency metrics, error rate dashboards, API usage patterns per consumer,
11. Cost of gateway as single point of failure — high availability patterns, gateway clustering, what happens when the gateway goes down



## 6. Service Mesh (App Mesh, Traffic Director, Anthos Service Mesh, Azure Service Mesh, Istio, Linkerd, Consul Connect, Cilium)
1. Sidecar proxy pattern — how Envoy/linkerd-proxy intercepts all pod traffic, iptables rules for transparent interception,
2. mTLS between services — automatic certificate rotation, SPIFFE/SPIRE identity framework, zero-trust networking,
3. Traffic splitting — canary deployments via weighted routing at mesh level, mirroring (shadow traffic) for testing,
4. Observability injection — automatic distributed tracing header propagation, L7 metrics without code changes,
5. Retry and timeout policies — per-route retry budgets, retry storms and cascading failures, retry budget as a percentage of normal traffic,
6. Circuit breaking and outlier detection — ejecting unhealthy pods based on consecutive 5xx, recovery probing,
7. Fault injection — injecting delays and errors for chaos engineering, testing circuit breakers in production,
8. Control plane vs data plane architecture — Istiod as control plane, Envoy sidecars as data plane, configuration propagation delay,
9. Performance overhead — latency added per hop (p99 sidecar cost), memory/CPU per sidecar, when the overhead isn't worth it,
10. Service mesh without sidecars — eBPF-based mesh (Cilium), ambient mesh (Istio waypoint proxies), trade-offs vs sidecar model



## 7. Service Discovery (Cloud Map, ECS Service Discovery, GKE/AKS built-in, Consul, etcd, ZooKeeper, Eureka)
1. Client-side vs server-side discovery — client queries registry directly vs LB resolves on behalf of client, pros/cons of each,
2. Self-registration vs third-party registration — service registers itself vs orchestrator registers it, coupling trade-offs,
3. Health check mechanisms — TTL-based heartbeats, active probing, script checks, what happens when checks are too aggressive or too slow,
4. DNS-based discovery — SRV records for port discovery, A record round-robin, TTL caching problems causing stale endpoints,
5. Consistency model for the registry — CP (Consul, ZooKeeper) vs AP (Eureka) — what happens during network partition,
6. Graceful deregistration — draining connections before removing from registry, timing coordination with health checks,
7. Multi-datacenter / multi-region discovery — WAN federation (Consul), cross-cluster service resolution, latency-aware routing,
8. Kubernetes native discovery — kube-dns/CoreDNS, Service and Endpoints objects, headless services for stateful sets



## 8. Distributed Coordination & Consensus (ZooKeeper, etcd, Consul, Redis Redlock, DynamoDB conditional writes, Cloud Spanner TrueTime)
1. Raft consensus deep dive — leader election, log replication, safety guarantees, what happens during split-brain,
2. Paxos vs Raft vs ZAB — trade-offs and practical differences, why Raft won in practice,
3. Distributed locking — fencing tokens to prevent stale lock holders from acting, why Redlock is controversial (Martin Kleppmann's analysis),
4. Leader election patterns — using ephemeral nodes (ZooKeeper), session-based leases (etcd), lease renewal and expiry semantics,
5. Watch/notification mechanisms — ZooKeeper watches (one-time) vs etcd watches (streaming), ordering guarantees,
6. Linearizability vs sequential consistency — what level of consistency does each system actually provide,
7. Split-brain scenarios — network partitions causing two leaders, how quorum-based systems prevent this, what happens with even-numbered clusters,
8. Performance under contention — write throughput limits of consensus systems, why they shouldn't be used as general-purpose databases,
9. TrueTime (Spanner) — using GPS+atomic clocks for globally consistent timestamps, commit-wait protocol



## 9. In-Memory Cache (ElastiCache Redis, ElastiCache Memcached, MemoryDB, Memorystore, Azure Cache for Redis, Redis, Memcached, Hazelcast, Dragonfly)
1. Cache-aside vs read-through vs write-through vs write-behind vs write-around — when to use each, consistency implications of each pattern,
2. Cache stampede / thundering herd — when TTL expires on hot key, all requests hit DB simultaneously, solutions: locking, probabilistic early expiry, external recomputation,
3. Hot key problem — single cache key receiving disproportionate traffic, solutions: key replication, local caching, key splitting,
4. Cache invalidation strategies — event-driven invalidation vs TTL-based, dual-delete pattern for write-through consistency,
5. Eviction policies deep dive — LRU vs LFU vs volatile-ttl vs allkeys-random, how Redis implements approximate LRU,
6. Redis persistence — RDB snapshots vs AOF (append-only file), fsync policies (always, everysec, no), recovery time vs data loss trade-offs,
7. Redis Cluster vs Sentinel — cluster mode (hash slots, resharding, cross-slot limitations) vs sentinel (HA for single-shard), when to use which,
8. Memory fragmentation — jemalloc behavior, activedefrag, memory overhead per key, how small objects waste memory,
9. Redis data structures beyond strings — sorted sets for leaderboards, HyperLogLog for cardinality, streams for event log, bitmaps for feature flags,
10. Consistent hashing for distributed cache — virtual nodes, minimizing cache misses during node add/remove,
11. Cache warming strategies — pre-populating cache after deploy/restart, shadow traffic, lazy loading trade-offs,
12. Memcached vs Redis — when Memcached's multi-threaded model wins (pure caching, simple strings, no persistence needed)



## 10. Relational Database (RDS, Aurora, Cloud SQL, AlloyDB, Azure SQL, Azure DB for MySQL/PostgreSQL, MySQL, PostgreSQL, MariaDB, CockroachDB, TiDB, YugabyteDB)
1. Isolation levels deep dive — read uncommitted → read committed → repeatable read → serializable, phantom reads, write skew anomalies, how PostgreSQL's MVCC implements each,
2. B-tree vs LSM-tree index internals — page splits, fill factor, write amplification in LSM, compaction strategies (leveled, tiered, FIFO),
3. Query optimizer internals — cost-based optimization, index selection, join ordering, EXPLAIN ANALYZE reading, common pitfalls (index not used, sequential scan),
4. Connection pooling — PgBouncer (transaction vs session vs statement mode), why unpooled connections kill performance at scale,
5. Replication topologies — async vs semi-sync vs synchronous, replication lag measurement, promoting replicas, split-brain avoidance,
6. Read replicas — eventual consistency challenges, routing reads vs writes, cross-region replicas for DR,
7. Sharding strategies — range vs hash vs directory-based, cross-shard queries, resharding pain, application-level vs proxy-level (Vitess, Citus),
8. Write-Ahead Log (WAL) — how WAL enables crash recovery and replication, WAL archiving, point-in-time recovery (PITR),
9. Vacuum and bloat (PostgreSQL) — dead tuple cleanup, autovacuum tuning, table bloat, transaction ID wraparound,
10. Deadlock detection and prevention — wait-for graphs, lock ordering, advisory locks, reducing lock contention,
11. Partitioning — range, hash, list partitioning, partition pruning, partition-wise joins, how partitioning differs from sharding,
12. Schema migration strategies — zero-downtime migrations, expand-and-contract pattern, online DDL (gh-ost, pt-online-schema-change),
13. N+1 query problem — detection, eager loading vs lazy loading, batching strategies, DataLoader pattern



## 11. Key-Value Store (DynamoDB, ElastiCache, Bigtable, Firestore, Cosmos DB Table API, Azure Table Storage, Redis, etcd, RocksDB, Riak, FoundationDB)
1. Partition key design — choosing keys for even distribution, avoiding hot partitions, composite keys for access pattern optimization,
2. Consistent hashing internals — virtual nodes, token ring, how DynamoDB/Cassandra distribute data across partitions,
3. Eventual consistency models — last-writer-wins (LWW), vector clocks, dotted version vectors, conflict resolution strategies,
4. Conditional writes and optimistic concurrency — DynamoDB ConditionExpression, compare-and-swap (CAS), preventing lost updates without locks,
5. LSM-tree storage engine — write path (memtable → immutable memtable → SSTable), read path (memtable → bloom filter → SSTable), compaction trade-offs,
6. DynamoDB capacity planning — on-demand vs provisioned, adaptive capacity, burst credits, throttling behavior, GSI/LSI capacity independence,
7. Global tables / multi-region replication — active-active replication, conflict resolution in DynamoDB global tables, replication lag,
8. Hot partition mitigation — write sharding (adding random suffix), caching hot reads, adaptive capacity redistribution,
9. TTL-based expiry — background deletion process, tombstones, how expired items affect reads before cleanup,
10. Single-table design (DynamoDB) — overloading partition/sort keys, GSI overloading, modeling 1:N and M:N relationships in a single table, access pattern-driven design



## 12. Document Database (DocumentDB, DynamoDB document mode, Firestore, MongoDB Atlas, Cosmos DB, MongoDB, CouchDB, RethinkDB, ArangoDB)
1. Schema design — embedding vs referencing, when to denormalize, bounded vs unbounded arrays, document size limits and growth patterns,
2. Indexing strategies — compound indexes and index prefix rule, multikey indexes on arrays, wildcard indexes, partial indexes, index intersection,
3. Aggregation pipeline optimization — pipeline ordering ($match early, $project to reduce), allowDiskUse for large datasets, $lookup performance (nested join),
4. Sharding — range vs hash shard keys, choosing shard key (cardinality, monotonicity, query patterns), jumbo chunks, rebalancing,
5. Replica set internals — election protocol, oplog tailing, write concern (w: majority) and read concern (majority, linearizable, snapshot),
6. Change streams — real-time event-driven processing, resume tokens, watch on collection/database/deployment, backpressure handling,
7. Transactions in MongoDB — multi-document ACID transactions, performance cost, transaction size limits, when to avoid them,
8. Read/write concern tuning — trade-off between durability (w: majority) and latency (w: 1), read preference (primary vs secondaryPreferred),
9. Data modeling anti-patterns — massive arrays, deep nesting, storing large blobs in documents, unnecessary indexing,
10. Schema validation — JSON Schema enforcement at DB level, validationAction (warn vs error), migration strategies for schema changes,
11. Time-series collections (MongoDB 5.0+) — bucketing, meta fields, automatic expiry, compression advantages over regular collections



## 13. Wide-Column Store (Keyspaces, DynamoDB wide-column, Bigtable, Cosmos DB Cassandra API, Azure Managed Cassandra, Cassandra, HBase, ScyllaDB)
1. Row key design — critical performance factor: composite keys, reverse timestamps for time-series, avoiding hotspots, key length trade-offs,
2. Column family design — grouping related columns, storage and compaction implications, wide rows vs tall tables,
3. LSM-tree and compaction strategies — size-tiered (STCS) vs leveled (LCS) vs time-window (TWCS), read/write amplification trade-offs,
4. Tunable consistency — consistency levels (ONE, QUORUM, LOCAL_QUORUM, ALL), read-repair, anti-entropy, hinted handoff,
5. Gossip protocol — peer-to-peer cluster membership, failure detection (phi accrual), seed nodes, how topology changes propagate,
6. Tombstones and deletion — why deletes are expensive, gc_grace_seconds, tombstone accumulation causing read latency spikes,
7. Time-series data modeling — TWCS compaction, bucketing by time period, TTL for automatic expiry, partition sizing (target <100MB),
8. Secondary indexes — limitations (scatter-gather queries), materialized views, SAI (Storage-Attached Indexes in Cassandra 4.0),
9. Repair and anti-entropy — full repair vs incremental repair, repair scheduling, what happens when repairs fall behind,
10. ScyllaDB vs Cassandra — shard-per-core architecture, seastar framework, when ScyllaDB's C++ implementation wins



## 14. Graph Database (Neptune, Neo4j Aura, Cosmos DB Gremlin API, Neo4j, JanusGraph, ArangoDB, Dgraph, TigerGraph)
1. Property graph vs RDF model — labeled property graphs (Neo4j) vs triple stores (RDF/SPARQL), when to use which,
2. Index-free adjacency — how Neo4j stores direct physical pointers between nodes, O(1) traversal vs index-based lookup,
3. Graph traversal optimization — BFS vs DFS trade-offs, traversal depth limits, pruning strategies, avoiding full graph scans,
4. Cypher vs Gremlin — declarative (Cypher) vs imperative (Gremlin) query patterns, performance implications,
5. Supernodes problem — nodes with millions of edges (celebrities in social graph), query performance degradation, partitioning strategies,
6. Graph partitioning — cutting edges vs replicating vertices, min-cut algorithms, why graph partitioning is fundamentally hard (NP-hard),
7. When graph DB beats relational joins — multi-hop traversals (friends-of-friends-of-friends), recursive queries, performance cliff in SQL with increasing join depth,
8. Knowledge graphs and ontologies — schema modeling with relationships, reasoning and inference, RDF/OWL,
9. Graph algorithms at scale — PageRank, community detection, shortest path, connected components, running on graph DBs vs batch frameworks (Pregel, GraphX)



## 15. Time-Series Database (Timestream, Bigtable, Azure Data Explorer/Kusto, Azure Time Series Insights, InfluxDB, TimescaleDB, Prometheus, VictoriaMetrics, QuestDB)
1. Compression algorithms — gorilla compression (XOR for timestamps, XOR for floats), delta-of-delta encoding, dictionary encoding for tags, achieving 1-2 bytes per data point,
2. Downsampling and retention policies — pre-aggregating old data (1s → 1m → 1h), continuous aggregation, tiered storage (hot → warm → cold),
3. High cardinality handling — why high-cardinality tags (user_id, request_id) destroy performance, inverted index blowup, solutions (limiting labels, pre-aggregation),
4. Write path optimization — batched writes, out-of-order write handling, WAL for durability, back-of-envelope: billions of data points per day,
5. Time-bucketing strategies — chunking data by time period for partition pruning and efficient deletion, chunk sizing trade-offs,
6. PromQL and query patterns — rate(), increase(), histogram_quantile(), subquery pitfalls, staleness handling,
7. Pull vs push model — Prometheus pull (service discovery + scraping) vs push (InfluxDB/VictoriaMetrics), trade-offs for ephemeral workloads,
8. Long-term storage — Thanos/Cortex/VictoriaMetrics for multi-cluster Prometheus, object storage backend, query federation



## 16. Search Engine / Full-Text Search (OpenSearch Service, CloudSearch, Elastic Cloud, Azure Cognitive Search, Elasticsearch, OpenSearch, Solr, Meilisearch, Typesense)
1. Inverted index internals — term dictionary, postings list, skip lists, how a query becomes a set intersection problem,
2. Analyzers and tokenization — character filters → tokenizer → token filters pipeline, custom analyzers, language-specific stemming, edge n-grams for autocomplete,
3. TF-IDF and BM25 scoring — how relevance is calculated, field boosting, function_score for custom ranking (recency, popularity),
4. Mapping and schema design — dynamic vs explicit mapping, keyword vs text fields, nested vs object, mapping explosion problem,
5. Sharding and replica strategy — shard sizing (target 10-50GB), too many shards problem (cluster state overhead), routing queries to specific shards,
6. Near-real-time (NRT) indexing — refresh interval, how segments are created, force merge for read performance, index lifecycle management (ILM),
7. Aggregations — terms, histogram, date_histogram, composite aggregations for pagination, cardinality (HyperLogLog), performance of nested aggregations,
8. Fuzzy matching and typo tolerance — edit distance (Levenshtein), n-gram based fuzzy, fuzziness=AUTO, performance cost of fuzzy queries,
9. Search relevance tuning — boosting, decay functions, script_score, multi_match (best_fields vs cross_fields vs phrase), A/B testing relevance models,
10. Elasticsearch as secondary index — syncing from primary database, handling consistency (lag), reindexing strategies with zero-downtime aliases



## 17. Vector Database (OpenSearch vector, MemoryDB vector, Vertex AI Vector Search, AlloyDB pgvector, Azure AI Search vector, Cosmos DB vector, Pinecone, Milvus, Weaviate, Qdrant, Chroma, pgvector, FAISS)
1. ANN algorithm internals — HNSW (navigable small world graphs), IVF (inverted file index), PQ (product quantization), trade-offs: recall vs latency vs memory,
2. Distance metrics — cosine similarity vs euclidean distance vs dot product, when to use each (normalized vs unnormalized embeddings),
3. Hybrid search — combining vector similarity with keyword/metadata filtering, pre-filter vs post-filter, performance implications,
4. Index tuning — HNSW parameters (M, efConstruction, efSearch), IVF nprobe, how these control recall-latency trade-off,
5. Quantization for compression — scalar quantization, product quantization, binary quantization, trading precision for memory (16x compression with binary),
6. Embedding dimension trade-offs — 384 vs 768 vs 1536 dimensions, storage cost, latency impact, dimensionality reduction techniques,
7. Sharding vectors — partitioning by metadata vs random, keeping semantically similar vectors together vs even distribution,
8. Real-time ingestion vs batch indexing — HNSW supports real-time inserts, IVF requires retraining centroids, hybrid approaches,
9. RAG architecture patterns — chunking strategies, embedding model selection, retrieval + reranking, hallucination reduction



## 18. Object Storage (S3, Cloud Storage, Blob Storage, MinIO, Ceph RADOS, OpenStack Swift, SeaweedFS)
1. Consistency model — S3 strong read-after-write consistency (since 2020), how this was achieved (witness system), eventual consistency in older APIs,
2. Storage classes and lifecycle — automatic tiering (hot → warm → cold → archive), retrieval latency and cost per class, minimum storage duration charges,
3. Multipart upload — parallel upload of large objects, part size tuning, completing/aborting multipart uploads, leftover parts cleanup,
4. Presigned URLs — secure time-limited access without exposing credentials, upload vs download presigned URLs, STS-based temporary credentials alternative,
5. Event notifications — triggering Lambda/Cloud Functions on PUT/DELETE, event ordering guarantees (or lack thereof), duplicate events,
6. Versioning and soft delete — protecting against accidental deletion, MFA delete, lifecycle rules for version cleanup, storage cost implications,
7. Cross-region replication — async replication, replication lag, conflict resolution (latest timestamp wins), bidirectional replication challenges,
8. Performance optimization — request parallelism, prefix-based partitioning (S3 auto-partitions now), byte-range fetches, transfer acceleration,
9. Server-side encryption — SSE-S3 vs SSE-KMS vs SSE-C, envelope encryption flow, key rotation, client-side encryption trade-offs,
10. Content-addressable storage — using content hash as key, deduplication, immutability guarantees, git-like storage model



## 19. Block Storage (EBS, GCP Persistent Disk, Azure Managed Disks, Ceph RBD, LVM, ZFS, OpenEBS)
1. IOPS vs throughput vs latency — understanding all three dimensions, how they interact, gp3 baseline vs burst vs io2 provisioned,
2. Snapshot mechanics — copy-on-write snapshots, incremental snapshots, cross-region snapshot copy, snapshot restore performance (lazy loading),
3. Multi-attach volumes — shared block storage for active-passive clusters, file system coordination requirements (GFS2, OCFS2),
4. Striping and RAID — RAID 0/1/5/6/10 trade-offs, software RAID (mdadm) vs hardware, how cloud providers abstract this away,
5. Volume resizing online — expanding volumes without downtime, filesystem resize (ext4/xfs), handling provisioned IOPS changes,
6. Placement groups — cluster placement (low latency) vs spread placement (high availability), impact on I/O performance,
7. Encryption at rest — dm-crypt/LUKS, cloud-managed encryption keys, performance overhead of encryption



## 20. File Storage (EFS, FSx Lustre/Windows/NetApp, GCP Filestore, Azure Files, Azure NetApp Files, NFS, GlusterFS, CephFS, Lustre, BeeGFS)
1. NFS vs SMB/CIFS — protocol differences, Linux vs Windows compatibility, NFS v3 vs v4 (stateful, Kerberos auth),
2. Throughput modes — bursting (credit-based) vs provisioned vs elastic, sizing for workload patterns,
3. POSIX compliance implications — file locking (advisory vs mandatory), permission model, how cloud file systems handle POSIX semantics,
4. Cross-AZ and cross-region — mount targets per AZ, data replication, performance degradation with distance,
5. High-performance parallel file systems — Lustre architecture (MDS + OSS + OST), scratch vs persistent, HPC and ML training workloads



## 21. Message Queue (SQS Standard/FIFO, Amazon MQ, GCP Cloud Tasks, Pub/Sub queue mode, Azure Queue Storage, Azure Service Bus, RabbitMQ, ActiveMQ, ZeroMQ, BullMQ)
1. Delivery guarantees — at-least-once vs at-most-once vs exactly-once (and why exactly-once is mostly a lie across system boundaries),
2. Visibility timeout tuning — too short = duplicate processing, too long = delayed retry on failure, dynamic extension during processing,
3. Dead-letter queues — configuring max receive count, monitoring DLQ depth, replaying DLQ messages, alerting strategies,
4. FIFO ordering guarantees — message group IDs for ordered processing per entity, throughput limitations of strict ordering,
5. Message deduplication — content-based deduplication vs explicit deduplication ID, deduplication window, idempotent consumers as complementary strategy,
6. Backpressure mechanisms — queue depth monitoring, auto-scaling consumers based on queue metrics, producer slowdown signals,
7. Poison pill handling — messages that repeatedly fail processing, isolation strategies, dead-lettering vs manual inspection,
8. Exactly-once processing patterns — idempotent consumers, transactional outbox pattern, deduplication at consumer side,
9. Priority queues — separate queues per priority vs priority field, starvation prevention for low-priority messages,
10. Fan-out with queues — SNS + SQS fan-out pattern, one event triggering multiple independent processing pipelines,
11. Batch processing — receiving/deleting messages in batches, partial batch failure handling, throughput optimization



## 22. Event Streaming Platform (Kinesis Data Streams, MSK, Pub/Sub, Dataflow, Event Hubs, Apache Kafka, Apache Pulsar, NATS JetStream, Redpanda)
1. Partition design — choosing partition key for even distribution and ordering, partition count planning (can't easily decrease), hot partitions,
2. Consumer groups — parallel consumption, partition assignment (range vs round-robin vs sticky), rebalancing storms and cooperative rebalancing,
3. Offset management — auto-commit risks, manual commit (at-least-once), exactly-once with transactions, consumer lag monitoring,
4. Log compaction — retaining only latest value per key, use cases (changelogs, materialized views), tombstones for deletion,
5. Exactly-once semantics (EOS) — idempotent producer, transactional producer, read_committed isolation, performance overhead,
6. ISR (In-Sync Replicas) — min.insync.replicas, acks=all behavior, unclean leader election trade-off (availability vs data loss),
7. Kafka producer tuning — batch.size, linger.ms, compression (snappy vs lz4 vs zstd), acks configuration, buffer.memory and backpressure,
8. Kafka consumer tuning — max.poll.records, max.poll.interval.ms, fetch.min.bytes, session.timeout.ms vs heartbeat.interval.ms,
9. Schema evolution with Schema Registry — backward vs forward vs full compatibility, Avro vs Protobuf, subject naming strategies,
10. Event sourcing pattern — storing events as source of truth, rebuilding state from event log, snapshot optimization, event versioning,
11. CQRS — separating read and write models, projecting events into read-optimized stores, eventual consistency implications,
12. Kafka Connect — source and sink connectors, exactly-once delivery, SMT (Single Message Transforms), distributed vs standalone mode,
13. Kafka Streams vs Flink — library vs framework, local state stores (RocksDB), KTable vs KStream, interactive queries



## 23. Pub/Sub (SNS, EventBridge, GCP Pub/Sub, Azure Service Bus Topics, Azure Event Grid, Kafka topics, NATS, Redis Pub/Sub, RabbitMQ exchanges)
1. Push vs pull delivery — push (webhook/HTTP endpoint) vs pull (consumer polls), at-least-once with acknowledgment, push endpoint reliability,
2. Message filtering — SNS filter policies (attribute-based), EventBridge content-based rules, filtering at broker vs consumer side,
3. Fan-out architectures — one event → many independent subscribers, each with own retry/DLQ, ordering per subscriber,
4. Dead-letter topics — messages that can't be delivered after max retries, monitoring and replay strategies,
5. Message ordering in pub/sub — Pub/Sub ordering keys, SNS FIFO topics, trade-offs with throughput,
6. Exactly-once delivery challenges — deduplication window, idempotent subscribers, acknowledge-before-process vs process-before-acknowledge,
7. EventBridge patterns — event bus architecture, schema discovery, archive and replay, cross-account/cross-region event routing



## 24. Stream Processing (Kinesis Data Analytics, Managed Flink, Dataflow/Apache Beam, Stream Analytics, Apache Flink, Spark Streaming, Kafka Streams)
1. Windowing semantics — tumbling vs sliding vs session vs global windows, window triggers (event time, processing time, count),
2. Watermarks and late data — how watermarks track event time progress, allowed lateness, side outputs for late events, watermark generation strategies,
3. Event time vs processing time — why event time matters (out-of-order events), ingestion time as compromise, skew handling,
4. State management — keyed state, operator state, state backends (RocksDB vs heap), state size and checkpointing overhead,
5. Checkpointing and exactly-once — Chandy-Lamport algorithm (Flink), checkpoint barriers, aligned vs unaligned checkpoints, checkpoint interval tuning,
6. Backpressure handling — Flink credit-based flow control, Spark micro-batch backpressure, monitoring and resolving bottlenecks,
7. Stream-table joins — enriching events with dimension data, temporal joins, slowly changing dimensions (SCD),
8. Savepoints and job versioning — taking savepoints for upgrades, state schema evolution, restoring from savepoints after code changes,
9. Exactly-once end-to-end — idempotent sinks, two-phase commit sinks (Kafka), at-least-once + dedup as alternative



## 25. Batch Processing (EMR, Glue, AWS Batch, Dataproc, Dataflow batch, HDInsight, Synapse Spark, Azure Batch, Apache Spark, Hadoop, Apache Beam, Dask)
1. Spark execution model — driver, executors, stages, tasks, DAG scheduler, how shuffles create stage boundaries,
2. Data skew handling — skewed joins (salting, broadcast join), skewed aggregations, monitoring via Spark UI (stage task times),
3. Shuffle optimization — shuffle partitions tuning, avoid unnecessary shuffles (broadcast join, co-partitioned data), external shuffle service,
4. Partitioning strategies — input partitioning (coalesce vs repartition), output partitioning (bucketing, partition columns), partition pruning,
5. Fault tolerance — lineage-based recovery (RDD), checkpoint vs recompute trade-offs, speculative execution for stragglers,
6. Resource management — YARN vs Kubernetes vs Mesos, dynamic allocation, executor sizing (memory vs cores),
7. Batch vs micro-batch vs streaming — Spark Structured Streaming micro-batch overhead, continuous processing mode, choosing batch interval,
8. Data formats — Parquet (columnar, predicate pushdown, statistics) vs Avro (row-based, schema evolution) vs ORC, compression (snappy, zstd, gzip),
9. Idempotent batch jobs — handling reruns safely, atomic output writes, partition overwrite vs append, deduplication



## 26. Workflow / Orchestration Engine (Step Functions, SWF, GCP Workflows, Cloud Composer, Logic Apps, Durable Functions, Temporal, Airflow, Cadence, Argo Workflows, Prefect, Dagster)
1. Saga pattern — orchestration (central coordinator) vs choreography (event-driven), compensating transactions for rollback, partial failure handling,
2. Temporal/Cadence deep dive — workflow-as-code, activity retries, signal and query, child workflows, workflow versioning (getVersion/patching),
3. Idempotency in workflows — activity idempotency tokens, handling duplicate workflow starts, deterministic workflow code requirements,
4. Long-running workflows — heartbeating for long activities, workflow timeouts, handling workflows that run for days/weeks,
5. Workflow versioning — deploying new workflow logic without breaking running instances, versioning strategies (Temporal patching, Step Functions aliases),
6. State machine design — AWS Step Functions states, choice/parallel/map states, error handling (Catch, Retry), execution history limits,
7. Airflow architecture — scheduler, workers, metadata DB, executor types (local, celery, kubernetes), DAG serialization, XCom pitfalls,
8. Retry policies — exponential backoff with jitter, max retries, non-retryable errors, timeout vs retry interaction



## 27. Task Scheduling / Cron (EventBridge Scheduler, Cloud Scheduler, Azure Functions Timer, Quartz, Celery Beat, Airflow)
1. Distributed cron — preventing duplicate execution when multiple instances are running, leader election for scheduler, distributed locking,
2. At-least-once execution — what happens when the scheduler fires but the task crashes, idempotent task design,
3. Time zone handling — storing schedules in UTC, DST transitions causing missed or double executions,
4. Backfill and catch-up — handling tasks that were missed during outage, Airflow catchup=True, ordering guarantees for backfill,
5. Job dependency chains — DAG-based scheduling, conditional execution, handling upstream failures



## 28. Serverless / FaaS (Lambda, Cloud Functions, Cloud Run, Azure Functions, OpenFaaS, Knative, Fission)
1. Cold start deep dive — JVM vs Node.js vs Python vs Go cold start times, provisioned concurrency, SNAPSTART (Lambda), warming strategies,
2. Concurrency models — reserved concurrency (throttling protection), provisioned concurrency (pre-warmed), per-function vs account-level limits,
3. Execution model — container reuse between invocations, /tmp persistence, connection pooling challenges (DB connections exhaustion),
4. Event source mapping — SQS, Kinesis, DynamoDB Streams as triggers, batch size and window, partial batch failure (bisect on error, report batch item failures),
5. Step Functions integration — orchestrating Lambdas, express vs standard workflows, cost at scale,
6. Cold start mitigation at scale — Lambda SnapStart (Java), provisioned concurrency auto-scaling, keep-warm hacks and why they're fragile,
7. Cost modeling — per-invocation + duration + memory pricing, break-even point vs always-on containers (typically ~1M requests/month),
8. VPC cold starts — ENI attachment delay, VPC-to-VPC Hyperplane, how AWS improved VPC Lambda cold starts dramatically



## 29. Container Orchestration (EKS, ECS, Fargate, GKE, Cloud Run, AKS, Azure Container Apps, Kubernetes, Docker Swarm, Nomad)
1. Pod scheduling deep dive — resource requests vs limits, QoS classes (Guaranteed, Burstable, BestEffort), scheduler predicates and priorities,
2. Horizontal Pod Autoscaler — CPU/memory-based scaling, custom metrics (Prometheus adapter), scaling behavior tuning (stabilization window, scaling policies),
3. Rolling deployment strategies — maxSurge and maxUnavailable, readiness gates, deployment rollback, how K8s detects failed deployments,
4. Persistent volumes — PV/PVC lifecycle, storage classes, dynamic provisioning, StatefulSet stable network identity and storage,
5. Network policies — pod-to-pod traffic control, ingress/egress rules, CNI plugins (Calico, Cilium), default-deny policies,
6. Ingress controllers — Nginx ingress, ALB ingress controller, path-based routing, TLS termination, canary annotations,
7. Resource limits and OOMKill — memory limits and OOMKilled pods, CPU throttling (CFS), right-sizing with VPA, resource quotas per namespace,
8. Health probes — liveness vs readiness vs startup probes, misconfigured probes causing restart loops, probe timing (initialDelaySeconds, periodSeconds),
9. Node affinity and taints/tolerations — dedicated node pools (GPU, high-memory), pod anti-affinity for HA, topology spread constraints,
10. Cluster autoscaler — scaling node pools based on pending pods, scale-down logic (underutilization threshold), node draining,
11. RBAC and multi-tenancy — namespace isolation, service accounts, role/clusterrole bindings, pod security standards



## 30. Compute / VMs (EC2, Compute Engine, Azure VMs, KVM, VMware, Proxmox, OpenStack Nova)
1. Instance type selection — compute vs memory vs storage vs GPU optimized families, burstable (T-series) credit system, right-sizing methodology,
2. Auto-scaling groups — scaling policies (target tracking, step, schedule-based), launch templates, instance refresh for rolling AMI updates,
3. Spot/preemptible instances — interruption handling (2-minute warning), spot fleet strategies, mixing on-demand + spot, checkpointing for fault tolerance,
4. Placement groups — cluster (low latency, high throughput), spread (max availability), partition (large distributed workloads like HDFS/Cassandra),
5. Machine images — AMI/snapshot lifecycle, baking images (Packer) vs bootstrapping (user-data), immutable infrastructure pattern,
6. Vertical scaling limitations — vCPU and memory ceilings, NUMA architecture, why horizontal scaling is preferred



## 31. Logging (CloudWatch Logs, OpenSearch, Cloud Logging, Azure Monitor Logs, ELK, EFK, Loki + Grafana, Graylog, Fluentd, Fluent Bit, Vector)
1. Structured logging design — JSON logs with consistent fields (trace_id, span_id, service, level, timestamp), avoiding free-text grepping at scale,
2. Log aggregation architecture — agent-based (Fluent Bit sidecar) vs daemonset, log pipeline (collect → buffer → route → store), backpressure handling,
3. Log sampling strategies — head-based sampling, tail-based (keep logs for errored requests), dynamic sampling based on error rate,
4. Correlation IDs — propagating request IDs across services, linking logs to traces, W3C Trace Context format,
5. Log retention and cost — hot/warm/cold tiering, log-based metrics (extracting numbers without storing raw logs), when to sample vs store all,
6. ELK vs Loki architecture — Elasticsearch (inverted index, full-text) vs Loki (label-based, chunks in object storage), cost and query trade-offs,
7. PII redaction — masking sensitive data in log pipeline, field-level redaction, compliance requirements (GDPR, HIPAA),
8. Log-based alerting — pattern matching on log streams, metric extraction from logs, avoiding alert fatigue from noisy logs



## 32. Metrics & Monitoring (CloudWatch Metrics, Managed Prometheus/Grafana, Cloud Monitoring, Azure Monitor, Prometheus + Grafana, VictoriaMetrics, Thanos, Cortex, Datadog)
1. RED and USE methods — RED (Rate, Errors, Duration) for request-driven services, USE (Utilization, Saturation, Errors) for resources, choosing the right method,
2. SLI/SLO/SLA framework — defining SLIs (latency p99 < 200ms), setting SLO (99.9%), error budget, error budget policies (freeze deploys when budget exhausted),
3. Percentiles vs averages — why p99 matters more than mean, coordinated omission problem, histogram vs summary metric types in Prometheus,
4. Cardinality management — high-cardinality labels (user_id, IP) causing metric explosion, label value guidelines, relabeling to drop cardinality,
5. Push vs pull collection — Prometheus pull (targets registered via service discovery) vs push (Pushgateway, VictoriaMetrics), trade-offs for ephemeral jobs,
6. Multi-cluster and long-term storage — Thanos (sidecar + store + compactor) vs Cortex (write path sharding) vs VictoriaMetrics, global query view,
7. Alerting on burn rate — multi-window burn rate alerts for SLO-based alerting, avoiding threshold-based alerts that fire too late or too often,
8. Dashboard design — avoiding vanity dashboards, debugging-oriented layouts, USE/RED templates, avoiding too many panels



## 33. Distributed Tracing (X-Ray, Cloud Trace, Application Insights, Jaeger, Zipkin, OpenTelemetry, Tempo, SigNoz)
1. OpenTelemetry architecture — SDK (instrumentation) → Collector (receive, process, export) → Backend, auto-instrumentation vs manual spans,
2. Trace context propagation — W3C TraceContext (traceparent header), B3 format, propagation across async boundaries (queues, cron),
3. Sampling strategies — head-based (decide at trace start) vs tail-based (decide after trace completes, keep errors/slow), probabilistic vs rate-limiting,
4. Span modeling — root span, child spans, span kind (client, server, producer, consumer), span attributes and events, span status,
5. Trace-log-metric correlation — linking trace IDs in logs, exemplars in Prometheus (trace ID on histogram buckets), unified observability,
6. Tail-based sampling architecture — collector aggregation layer, partial trace assembly, memory pressure from holding spans, ParentBased sampling,
7. Instrumentation of async flows — tracing across Kafka (produce → consume), tracing through queues, context injection in message headers



## 34. Alerting & Incident Management (CloudWatch Alarms, Cloud Monitoring Alerting, Azure Monitor Alerts, Alertmanager, Grafana Alerting, PagerDuty, OpsGenie)
1. Alert fatigue reduction — actionable alerts only, reducing noise via grouping, inhibition, silencing, alert ownership,
2. Multi-window burn rate alerts — fast-burn (short window, high threshold) + slow-burn (long window, low threshold) for SLO-based alerting,
3. Escalation policies — primary → secondary → engineering manager → VP chain, time-based escalation, business hours routing,
4. Runbooks and automation — linking alerts to runbooks, automated remediation (auto-restart, auto-scale), human-in-the-loop for destructive actions,
5. Alertmanager architecture — grouping (group_by labels), inhibition (suppress dependent alerts), silencing (maintenance windows), routing tree,
6. Post-mortem process — blameless post-mortems, timeline reconstruction, action items tracking, severity classification



## 35. Authentication & Authorization (Cognito, IAM, STS, Identity Platform, Cloud IAM, Identity-Aware Proxy, Entra ID, Azure AD B2C, Keycloak, Auth0, Ory, Dex)
1. OAuth 2.0 flows — authorization code + PKCE (SPAs, mobile), client credentials (service-to-service), device code flow, when to use which,
2. JWT deep dive — header.payload.signature structure, RS256 vs HS256, token size implications, claims (iss, sub, aud, exp), JWT validation without calling auth server,
3. Token lifecycle — short-lived access tokens + refresh tokens, token rotation, revocation challenges with stateless JWT, token introspection endpoint,
4. RBAC vs ABAC vs ReBAC — role-based (simple) vs attribute-based (flexible) vs relationship-based (Zanzibar/Google), choosing for your scale,
5. Session management — server-side sessions (Redis-backed) vs stateless JWT, session fixation, concurrent session limits,
6. Service-to-service authentication — mTLS, workload identity (K8s service accounts → cloud IAM), SPIFFE, JWT for internal services,
7. MFA implementation — TOTP (Google Authenticator), WebAuthn/FIDO2 (passkeys), backup codes, adaptive MFA (risk-based),
8. Google Zanzibar — relationship-based access control at scale, tuple stores, check API, expand API, how Authzed/SpiceDB implement it



## 36. Secrets Management (Secrets Manager, Parameter Store, KMS, GCP Secret Manager, Cloud KMS, Azure Key Vault, HashiCorp Vault, SOPS, Sealed Secrets, Infisical)
1. Envelope encryption — data key encrypts data, master key encrypts data key, why this two-layer approach (key rotation, HSM constraints),
2. Dynamic secrets — generating short-lived database credentials on-demand (Vault), automatic revocation, reducing blast radius of credential leak,
3. Secret rotation — zero-downtime rotation patterns (dual-secret strategy), Lambda-based rotation (Secrets Manager), rotation interval trade-offs,
4. Vault architecture — seal/unseal process, Shamir's Secret Sharing for unseal keys, auto-unseal with cloud KMS, storage backends,
5. Secret injection patterns — environment variables (12-factor but visible in process table) vs mounted files vs init container vs sidecar, CSI Secrets Store driver,
6. Audit logging — every secret access logged, detecting anomalous access patterns, compliance requirements



## 37. WAF (AWS WAF, Shield, Cloud Armor, Azure WAF, ModSecurity, Coraza, NAXSI)
1. OWASP Core Rule Set — SQLi, XSS, RCE, LFI/RFI detection rules, false positive tuning (paranoia levels), rule exclusion patterns,
2. Rate limiting rules — per-IP, per-URI, per-session rate rules, distributed rate limiting across edge nodes,
3. Bot mitigation — distinguishing good bots (Googlebot) from bad, CAPTCHA challenges, JavaScript challenges, behavioral analysis,
4. Custom rules — pattern matching on headers/body/URI, regex-based rules, IP reputation lists, geo-blocking,
5. False positive management — logging-only mode before blocking, tuning rules for specific application paths, whitelisting trusted sources,
6. WAF placement — at CDN edge (CloudFront + WAF) vs at load balancer (ALB + WAF), latency and coverage trade-offs



## 38. DDoS Protection (Shield Standard/Advanced, Cloud Armor, Azure DDoS Protection, Cloudflare, Akamai Prolexic)
1. Attack types — volumetric (UDP flood, DNS amplification), protocol (SYN flood, Ping of Death), application-layer (HTTP flood, slow POST),
2. Anycast-based mitigation — distributing attack traffic across global PoPs, absorbing volumetric attacks at the edge,
3. Traffic scrubbing — rerouting through scrubbing centers, clean traffic forwarded to origin, BGP-based rerouting,
4. Application-layer DDoS — L7 attacks that look like legitimate traffic, rate limiting, challenge-response, behavioral fingerprinting,
5. Always-on vs on-demand — always-on (Cloudflare, Shield Standard) vs on-demand scrubbing (Prolexic, Shield Advanced), detection-to-mitigation time



## 39. Encryption & PKI (KMS, ACM, CloudHSM, Cloud KMS, CA Service, Key Vault, Azure Dedicated HSM, Let's Encrypt, cert-manager, OpenSSL, step-ca)
1. TLS handshake deep dive — ClientHello → ServerHello → certificate → key exchange → finished, TLS 1.3 simplification (1-RTT), 0-RTT resumption risks,
2. mTLS — both client and server present certificates, certificate validation chain, use in service mesh, bootstrapping trust,
3. Certificate lifecycle — issuance, renewal, revocation (CRL vs OCSP vs OCSP stapling), automated renewal (cert-manager, ACM),
4. HSM vs software KMS — FIPS 140-2 levels, when HSM is required (compliance, key ceremony), latency implications,
5. Envelope encryption flow — generate DEK → encrypt data with DEK → encrypt DEK with KEK in KMS → store encrypted DEK with data,
6. Key rotation — rotating KEK without re-encrypting all data (envelope encryption benefit), rotating DEK, key versioning,
7. Client-side encryption — encrypting before sending to storage, key management complexity, when required (zero-trust data, regulatory)



## 40. Data Warehouse (Redshift, BigQuery, Synapse Analytics, ClickHouse, Apache Druid, Apache Pinot, DuckDB, Trino, Greenplum)
1. Columnar storage internals — column chunks, encoding (dictionary, RLE, bit-packing), predicate pushdown, late materialization,
2. Star schema vs snowflake — dimension tables, fact tables, denormalization trade-offs, slowly changing dimensions (SCD type 1/2/3),
3. Separation of storage and compute — BigQuery architecture, Redshift Serverless, scaling compute independently, caching across restarts,
4. Partitioning and clustering — partition pruning for time-based queries, clustering for frequently filtered columns, partition granularity trade-offs,
5. Materialized views — pre-computed aggregations, incremental refresh, stale data trade-offs, cost of maintenance,
6. Query optimization — columnar scan vs index scan, join algorithms (hash, sort-merge, broadcast), query execution plan analysis,
7. Data lakehouse pattern — Delta Lake/Iceberg on object storage with warehouse-like performance, ACID on lakes, time-travel queries,
8. ClickHouse internals — MergeTree engine family, sparse primary index, materialized views as real-time aggregation, ReplacingMergeTree for dedup



## 41. Data Lake (S3 + Glue + Athena, Lake Formation, Cloud Storage + BigQuery, Dataplex, ADLS Gen2, Synapse serverless, Apache Iceberg, Delta Lake, Apache Hudi, Hive Metastore)
1. Schema-on-read vs schema-on-write — flexibility vs governance, how schema enforcement is added back via table formats,
2. Table formats — Iceberg vs Delta Lake vs Hudi, snapshot isolation, time-travel, schema evolution, partition evolution (Iceberg),
3. File format selection — Parquet (columnar, analytics) vs Avro (row-based, streaming) vs ORC, compression codec selection,
4. Small files problem — too many small files killing query performance, compaction strategies, file sizing targets (128MB–1GB),
5. Data catalog and governance — Glue Catalog, Hive Metastore, Unity Catalog, data lineage, data classification, access control,
6. Partitioning strategy — physical partitioning (directory-based), partition pruning, over-partitioning vs under-partitioning, Iceberg hidden partitioning,
7. ACID transactions on data lakes — how Iceberg/Delta achieve atomicity (metadata-based), concurrent writes, optimistic concurrency,
8. Lakehouse architecture — combining data lake (raw storage) + warehouse (structured query) in one system, eliminating ETL between lake and warehouse



## 42. ETL / ELT & Data Integration (Glue, DMS, Dataflow, Dataproc, Data Fusion, Data Factory, Synapse Pipelines, Airflow, dbt, NiFi, Airbyte, Debezium, Meltano)
1. ETL vs ELT trade-offs — transform before load (ETL: cleaner warehouse) vs load then transform (ELT: leverage warehouse compute power),
2. Incremental processing — change detection (timestamps, CDC, watermarks), merge (upsert) strategies, handling late-arriving data,
3. dbt deep dive — SQL-based transformations, ref() for dependency management, testing (unique, not_null, custom), documentation generation, incremental models,
4. Idempotent pipelines — designing transformations that produce the same result on re-run, partition overwrite vs merge, deduplication,
5. Schema evolution handling — adding/removing columns, type changes, forward/backward compatibility, schema migration strategies,
6. Backfill strategies — re-processing historical data without disrupting current pipeline, partition-based backfill, time-bounded re-runs,
7. Data quality checks — Great Expectations, dbt tests, data freshness monitoring, data contracts between producers and consumers,
8. Pipeline orchestration patterns — Airflow DAGs, dependency management, sensor operators (waiting for data), dynamic DAG generation



## 43. Change Data Capture (DMS, DynamoDB Streams, Kinesis, GCP Datastream, Cosmos DB Change Feed, Debezium, Maxwell, Canal)
1. Log-based CDC — tailing database WAL/binlog, no application code changes, minimal overhead on source DB, supported databases,
2. Initial snapshot + streaming — bootstrapping the CDC pipeline with full table snapshot, then switching to incremental changes, consistency during transition,
3. Outbox pattern — writing events to an outbox table in the same transaction as business data, CDC captures outbox, guarantees exactly-once publish,
4. Schema evolution in CDC — handling ALTER TABLE during streaming, Debezium schema history topic, breaking changes vs backward-compatible,
5. Ordering guarantees — per-key ordering, how partition key in Kafka aligns with primary key in source, handling multi-table transactions,
6. Tombstone events — representing DELETE as null-value message, compaction interaction, downstream consumer handling,
7. Debezium architecture — Kafka Connect source connector, snapshot mode (initial, schema_only, never), slot management (PostgreSQL replication slots),
8. Replication slot management — PostgreSQL WAL retention when slot falls behind, disk space risk, monitoring slot lag



## 44. Notification Service (SNS push, SES email, Pinpoint, Firebase Cloud Messaging, Notification Hubs, Azure Communication Services, Novu, ntfy, SendGrid, Twilio)
1. Multi-channel orchestration — deciding which channel (push, email, SMS, in-app) based on urgency and user preference, channel fallback chains,
2. Delivery guarantees per channel — push notification delivery is best-effort (device offline), email bounce handling, SMS delivery receipts,
3. User preference management — opt-in/opt-out per channel, quiet hours, frequency capping (max N notifications per hour),
4. Push notification architecture — APNS (Apple) and FCM (Google) flows, device token management, token invalidation, silent push for background updates,
5. Template management — localization, personalization variables, A/B testing notification content, rich notifications (images, actions),
6. Rate limiting and throttling — respecting provider rate limits (SES sending quotas), self-imposed limits to prevent notification fatigue,
7. Delivery analytics — open rates, click-through rates, unsubscribe tracking, attribution



## 45. Rate Limiter (API Gateway throttling, WAF rate rules, Cloud Armor, Apigee quotas, Azure API Management, Redis + Lua, Nginx limit_req, Envoy, resilience4j)
1. Token bucket algorithm — bucket capacity (burst), refill rate (sustained), implementation with timestamps (avoid timer threads),
2. Sliding window log — storing each request timestamp, precise but memory-expensive, Redis sorted set implementation,
3. Sliding window counter — hybrid of fixed window and sliding log, weighted count from previous and current window, good accuracy with low memory,
4. Fixed window problems — boundary burst (2x rate at window boundary), simple but imprecise,
5. Distributed rate limiting — centralized counter in Redis, race conditions with GET+SET (use Lua scripts or MULTI/EXEC), eventual consistency trade-off,
6. Client identification — rate limit by API key, user ID, IP, or combination, handling proxies and shared IPs,
7. Hierarchical rate limits — global → per-service → per-user → per-endpoint limits, which layer enforces what,
8. Graceful handling — 429 response with Retry-After header, response headers (X-RateLimit-Remaining, X-RateLimit-Reset), client-side backoff



## 46. Circuit Breaker (App Mesh, Traffic Director, Cloud Service Mesh, Azure Service Mesh, Hystrix, resilience4j, Polly, Envoy, Istio)
1. State machine — closed (normal) → open (fail-fast) → half-open (probe), transition thresholds and timers,
2. Failure detection — error count vs error percentage, slow call detection (latency threshold), which errors count (5xx yes, 4xx no),
3. Half-open state — allowing limited probe requests, success threshold to close, failure in half-open immediately re-opens,
4. Fallback strategies — cached response, default value, degraded functionality, queue-for-later, how fallbacks compose in call chains,
5. Bulkhead pattern — isolating thread pools or connection pools per dependency, preventing one slow service from exhausting all resources,
6. Retry + circuit breaker interaction — retries with exponential backoff + jitter, retry budget limiting total retry traffic, retries trigger circuit breaker faster,
7. Circuit breaker in distributed systems — per-instance vs shared state circuit breakers, cascading circuit breaker opening across service graph,
8. Monitoring circuit state — metrics for open/closed/half-open transitions, dashboards, alerts when circuits open frequently



## 47. Configuration Management (AppConfig, Parameter Store, Runtime Configurator, Firebase Remote Config, App Configuration, Consul KV, etcd, Spring Cloud Config, LaunchDarkly, Unleash, Flagsmith)
1. Feature flags deep dive — boolean flags, multivariate flags, user-targeted flags, percentage rollouts, flag lifecycle (create → enable → clean up),
2. Dark launches — deploying code behind disabled flags, enabling for internal users first, monitoring before public rollout,
3. Gradual rollouts — 1% → 10% → 50% → 100%, automatic rollback on error rate spike, sticky bucketing (same user always gets same variant),
4. Configuration propagation — polling vs push vs long-polling, propagation delay, consistency (what if half the fleet has new config),
5. Kill switches — emergency feature disabling without deploy, predefining kill switches for risky features,
6. Configuration validation — schema validation before push, canary config deployment, preventing typos from breaking production,
7. Static vs dynamic config — what belongs in deploy artifacts vs runtime config store, security implications of runtime override



## 48. Content Management / Blob Processing (S3 + Lambda + MediaConvert + CloudFront, Cloud Storage + Cloud Functions + Transcoder API, Blob Storage + Functions + Media Services, MinIO + FFmpeg, Tus protocol)
1. Resumable upload — Tus protocol, multipart upload with checkpointing, handling unreliable mobile connections,
2. Processing pipeline design — event-driven (S3 event → Lambda/queue → processor), fan-out for multiple output formats, pipeline ordering,
3. Virus scanning — ClamAV integration, scanning before serving, quarantine bucket pattern, async scanning with status field,
4. Content-addressable storage — hash-based keys (SHA-256), deduplication, immutability, Git-like content storage,
5. Thumbnail and preview generation — on-upload (eager) vs on-first-request (lazy), pre-signed URL for generated assets, cache invalidation,
6. Content moderation — integrating AI moderation (Rekognition, Cloud Vision), pre-publish review queue, appeal workflow



## 49. ID Generation (DynamoDB auto IDs, Firestore auto IDs, Spanner unique timestamps, Twitter Snowflake, Sonyflake, ULID, UUID, KSUID)
1. UUID v4 vs UUID v7 (time-ordered) — randomness vs sortability, database index fragmentation from random UUIDs, why v7 matters for B-tree performance,
2. Snowflake ID design — 41-bit timestamp + 10-bit machine ID + 12-bit sequence, clock skew handling (wait until next millisecond), machine ID assignment,
3. ULID — lexicographically sortable, Crockford's Base32, monotonic sort order within same millisecond, compatibility with UUID storage,
4. Collision probability — birthday problem math, UUID v4 collision odds, why 128-bit randomness is sufficient for most systems,
5. ID as partition key — sortable IDs cause hot partition on latest partition, random IDs spread writes but lose time-ordering, compromise strategies,
6. Database auto-increment at scale — single point of failure, multi-master conflicts (odd/even), block allocation (Flickr ticket server),
7. K-sortable IDs — KSUID (128-bit, seconds resolution), benefits for database indexing, time-extraction from ID, privacy implications



## 50. Email Service (SES, Azure Communication Services Email, Postfix, SendGrid, Mailgun, Postal)
1. Email authentication — SPF (authorized senders), DKIM (cryptographic signature), DMARC (policy enforcement), alignment requirements,
2. Bounce handling — hard bounce (invalid address, remove immediately) vs soft bounce (mailbox full, retry), feedback loops (complaints),
3. Sending reputation — dedicated vs shared IP, IP warm-up schedule (start low, increase daily), domain reputation, sender score monitoring,
4. Deliverability optimization — authentication passing, list hygiene, engagement tracking, avoiding spam trigger words (minor factor), proper unsubscribe headers,
5. Transactional vs marketing — separate streams (different IPs/domains), CAN-SPAM/GDPR compliance, unsubscribe handling,
6. Email infrastructure at scale — MTA architecture, queue management, retry scheduling, connection pooling to receiving mail servers



## 51. Media Processing (MediaConvert, MediaLive, Elastic Transcoder, Transcoder API, Video Intelligence API, Media Services, FFmpeg, Sharp, Imgproxy, Thumbor)
1. Adaptive bitrate streaming — HLS vs DASH, manifest files (m3u8/mpd), segment duration trade-offs, multi-rendition encoding,
2. Codec selection — H.264 (broad compatibility) vs H.265/HEVC (50% smaller) vs VP9 vs AV1 (best compression, slow encoding), browser support matrix,
3. Transcoding pipeline — job queue architecture, parallel encoding for multiple renditions, priority queues for live vs VOD,
4. Image optimization — format selection (WebP for browsers, AVIF for modern, JPEG fallback), quality-size curve, responsive images (srcset),
5. DRM (Digital Rights Management) — Widevine, FairPlay, PlayReady, license server architecture, key rotation, offline playback,
6. Live streaming architecture — ingest (RTMP/SRT) → transcoding → packaging → CDN → player, latency tiers (standard 20-30s, low <5s, ultra-low <1s)



## 52. Geospatial / Location Service (AWS Location Service, Google Maps Platform, BigQuery GIS, Azure Maps, PostGIS, OpenStreetMap, OSRM, H3)
1. Geohashing — encoding lat/lng into hierarchical string, prefix-based proximity search, edge-case at geohash boundaries (neighbors),
2. Spatial indexing — R-tree (range queries), quadtree (recursive subdivision), K-D tree, PostGIS GIST indexes, when to use which,
3. Proximity search at scale — geohash prefix scan, bounding box + Haversine refinement, H3 hexagonal grid (Uber) for uniform area cells,
4. Geofencing — point-in-polygon algorithms, real-time geofence triggering, scaling to millions of geofences,
5. Route optimization — Dijkstra vs A* vs contraction hierarchies, real-time traffic integration, multi-stop route optimization (TSP approximation),
6. PostGIS deep dive — geometry vs geography types, ST_DWithin for proximity, spatial joins, index-only scans on spatial data



## 53. Real-Time Communication (API Gateway WebSocket, AppSync subscriptions, IoT Core, Firebase RTDB, Cloud Run WebSocket, SignalR, Web PubSub, Socket.IO, Centrifugo, Mercure)
1. WebSocket vs SSE vs long polling — bidirectional (WebSocket) vs server-push (SSE), reconnection semantics, HTTP/2 compatibility,
2. Horizontal scaling WebSockets — sticky sessions vs pub/sub fan-out (Redis, Kafka), connection state in stateless infrastructure,
3. Connection management at scale — heartbeat/ping-pong, idle connection cleanup, memory per connection, 100K+ concurrent connections per node,
4. Presence detection — tracking who is online, distributed presence with Redis pub/sub, heartbeat-based timeout, presence at scale (millions of users),
5. Message ordering — per-channel ordering, global ordering challenges, vector clocks for conflict resolution,
6. Reconnection and offline sync — client-side message queue during disconnect, last-event-ID (SSE), cursor-based catch-up on reconnect,
7. Fan-out architecture — one publish → millions of subscribers (live sports, stock tickers), tiered fan-out, edge fan-out



## 54. Infrastructure as Code (CloudFormation, CDK, GCP Deployment Manager, Config Connector, ARM Templates, Bicep, Terraform, OpenTofu, Pulumi, Ansible, Crossplane)
1. Declarative vs imperative — Terraform/CloudFormation (declare desired state) vs Pulumi/CDK (write in programming language), trade-offs,
2. State management — Terraform state file, remote backends (S3 + DynamoDB lock), state locking, state file corruption recovery,
3. Drift detection — detecting manual changes (terraform plan), reconciliation strategies, preventing drift (RBAC, CI-only deploys),
4. Module design — reusable modules, input/output contracts, module versioning, module registry, mono-module vs multi-module,
5. Secret handling in IaC — avoiding secrets in state file, referencing Vault/Secrets Manager, encrypted state backends,
6. Testing IaC — unit tests (Terratest), plan validation, policy-as-code (OPA, Sentinel, Checkov), preview environments,
7. Import existing resources — terraform import, adoptive resource management, gradual IaC adoption strategy



## 55. CI/CD Pipeline (CodePipeline, CodeBuild, CodeDeploy, Cloud Build, Cloud Deploy, Azure DevOps Pipelines, GitHub Actions, GitLab CI, Jenkins, Argo CD, Tekton)
1. Deployment strategies — rolling (incremental replace), blue-green (instant switch), canary (gradual traffic shift), shadow/dark deployment,
2. Rollback mechanisms — automatic rollback on health check failure, database rollback challenges, forward-fix vs rollback culture,
3. Build caching — layer caching (Docker), dependency caching, remote build caches, cache invalidation,
4. GitOps — Argo CD / Flux CD, declarative desired state in git, pull-based deployment, drift reconciliation,
5. Trunk-based development — short-lived feature branches, feature flags for incomplete features, release branching anti-patterns,
6. Pipeline security — SAST/DAST/SCA scanning, secret scanning, signed artifacts, supply chain security (SLSA, Sigstore),
7. Environment promotion — dev → staging → production pipeline, environment parity, promotion gates (manual approval, automated tests),
8. Artifact management — immutable artifact versioning, container image tagging strategies, artifact promotion vs rebuild



## 56. Container Registry (ECR, Artifact Registry, Azure Container Registry, Docker Hub, Harbor, Quay, Nexus, GHCR)
1. Image tagging strategies — avoid :latest in production, semantic versioning, git SHA tags, immutable tags, tag retention policies,
2. Vulnerability scanning — Trivy, Grype, Snyk, scanning in CI vs registry-side scanning, blocking vulnerable images from deploy,
3. Image signing and verification — cosign (Sigstore), Notary v2, admission controllers that verify signatures, supply chain trust,
4. Multi-architecture images — manifest lists, building for amd64 + arm64 (Graviton), buildx cross-compilation,
5. Layer caching and deduplication — how registries deduplicate shared layers, pull optimization, pre-pulling on nodes,
6. Garbage collection — deleting untagged manifests, retention policies, registry storage costs



## 57. Event Router (EventBridge, Eventarc, Event Grid, Kafka + Kafka Streams, Apache Camel, RabbitMQ routing keys, NATS)
1. Rule-based routing — matching event patterns to targets, content-based filtering, event schema enforcement,
2. Event schema evolution — adding/removing fields without breaking consumers, schema registry integration, event versioning strategies,
3. Archive and replay — storing all events, replaying from a point in time for debugging or rebuilding state, event store as source of truth,
4. Cross-account/cross-region routing — federated event buses, routing events between AWS accounts or regions, security (event bus policies),
5. Dead-letter handling — events that fail delivery after retries, monitoring dead-letter targets, retry and replay from DLQ,
6. Event bus architecture — central bus vs topic-per-event-type, discovery of event types, coupling trade-offs



## 58. Data Serialization / Schema Registry (Glue Schema Registry, MSK Schema Registry, Pub/Sub schemas, Azure Schema Registry, Confluent Schema Registry, Apicurio, Buf)
1. Avro vs Protobuf vs JSON Schema — Avro (schema in file, compact binary), Protobuf (field numbering, backward compatible by default), JSON Schema (human-readable but large),
2. Compatibility levels — backward (new reader, old data), forward (old reader, new data), full (both), transitive variants, breaking vs non-breaking changes,
3. Schema evolution strategies — Avro (add field with default), Protobuf (new field number, never reuse), JSON (additive changes), handling required fields,
4. Schema registry in event-driven architecture — producer registers schema, consumer fetches schema by ID, schema caching, version negotiation,
5. Protobuf + gRPC integration — proto file as contract, code generation, backward-compatible service evolution, buf lint/breaking checks,
6. Serialization performance — binary formats (Protobuf, Avro, FlatBuffers) vs text (JSON, XML), encoding/decoding speed, payload size comparison