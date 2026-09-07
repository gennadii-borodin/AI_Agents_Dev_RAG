# Hybrid RAG — Bank App Requirements Knowledge Base

Demonstrates hybrid retrieval (vector similarity + graph traversal) over a bank mobile app requirements corpus.

## Language

**Business Requirement (BR)**:
A high-level capability the bank app must provide to users.
_Avoid_: business rule, functional spec

**Functional Requirement (FR)**:
A concrete system behavior that implements one business requirement. Always references a single BR via IMPLEMENTS.
_Aavoid_: feature, spec

**Non-Functional Requirement (NFR)**:
A quality constraint (performance, security, compatibility) that applies to one business requirement. Always references a single BR via IMPLEMENTS.
_Avoid_: quality attribute, constraint

**Test Scenario (TS)**:
A concrete test case that verifies one functional or non-functional requirement. Always references a single FR or NFR via COVERS.
_Avoid_: test case, use case

**Knowledge Graph**:
A Neo4j graph where requirement and test nodes are linked by IMPLEMENTS and COVERS relationships.
_Avoid_: graph database, KG

**Vector Store**:
A ChromaDB collection storing embeddings of all requirement and test texts for semantic similarity search.
_Avoid_: embedding store, vector DB

**Hybrid Retrieval**:
A two-phase query strategy: vector search finds semantically relevant nodes, then graph traversal expands results via IMPLEMENTS/COVERS relationships.
_Avoid_: combined search, merged retrieval
