-- 0001_init: documents and chunks, with dense (HNSW) and lexical (tsvector/GIN) indexes.
--
-- Forward-only. Applied inside a single transaction by writeoff.db.migrate.
-- The embedding dimension (1024) must match writeoff.config.EMBEDDING_DIMENSION.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id              uuid PRIMARY KEY,
    source_url      text        NOT NULL,
    title           text        NOT NULL,
    doc_type        text        NOT NULL
        CHECK (doc_type IN ('irc', 'treasury_regulation', 'irs_publication', 'form_instructions')),
    tax_year        smallint    NOT NULL CHECK (tax_year BETWEEN 2000 AND 2100),
    effective_date  date,
    retrieved_at    timestamptz NOT NULL,
    content_hash    char(64)    NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    -- A new tax year's edition of a publication is a new document, never an overwrite.
    UNIQUE (source_url, tax_year)
);

CREATE TABLE chunks (
    id               uuid PRIMARY KEY,
    document_id      uuid        NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    parent_id        uuid                 REFERENCES chunks (id)    ON DELETE CASCADE,
    level            text        NOT NULL CHECK (level IN ('parent', 'child')),
    ordinal          integer     NOT NULL CHECK (ordinal >= 0),

    citation_path    text        NOT NULL CHECK (citation_path <> ''),
    breadcrumb       text        NOT NULL,
    text             text        NOT NULL,          -- raw passage, for display and citation
    context_summary  text,                          -- contextual-retrieval summary
    token_count      integer     NOT NULL CHECK (token_count > 0),
    content_hash     char(64)    NOT NULL,

    -- Denormalized from documents so filters need no join on the hot search path.
    source_url       text        NOT NULL,
    title            text        NOT NULL,
    doc_type         text        NOT NULL
        CHECK (doc_type IN ('irc', 'treasury_regulation', 'irs_publication', 'form_instructions')),
    tax_year         smallint    NOT NULL CHECK (tax_year BETWEEN 2000 AND 2100),
    effective_date   date,
    retrieved_at     timestamptz NOT NULL,
    entity_types     text[]      NOT NULL DEFAULT '{}',   -- empty = applies to all entities

    -- Only child chunks are embedded and searched; parents are returned for context.
    embedding        vector(1024),

    -- Lexical search. citation_path uses the 'simple' config so tokens like "280a" and
    -- "8829" are kept verbatim; prose uses 'english' for stemming. Weights let exact
    -- citation matches outrank passing mentions.
    search_tsv       tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple',  citation_path), 'A') ||
        setweight(to_tsvector('english', breadcrumb),    'B') ||
        setweight(to_tsvector('english', text),          'C')
    ) STORED,

    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT chunks_parent_iff_child CHECK ((level = 'child') = (parent_id IS NOT NULL)),
    CONSTRAINT chunks_child_size       CHECK (level = 'parent' OR token_count <= 1200),
    CONSTRAINT chunks_embedding_level  CHECK (level = 'child' OR embedding IS NULL)
);

-- Dense retrieval: HNSW with cosine distance. m=16 / ef_construction=64 are pgvector's
-- defaults and a sound starting point for a corpus in the tens of thousands of chunks;
-- tune alongside hnsw.ef_search in the retrieval evals (phase 4).
CREATE INDEX chunks_embedding_hnsw_idx ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Lexical retrieval.
CREATE INDEX chunks_search_tsv_gin_idx ON chunks USING gin (search_tsv);

-- Metadata filters and lookups.
CREATE INDEX chunks_year_type_level_idx ON chunks (tax_year, doc_type, level);
CREATE INDEX chunks_entity_types_gin_idx ON chunks USING gin (entity_types);
CREATE INDEX chunks_citation_idx ON chunks (tax_year, citation_path text_pattern_ops);
CREATE INDEX chunks_document_idx ON chunks (document_id);
CREATE INDEX chunks_parent_idx ON chunks (parent_id) WHERE parent_id IS NOT NULL;
