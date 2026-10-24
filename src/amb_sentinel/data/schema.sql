-- AMB-Sentinel schema. This is the ONLY place DDL lives.
-- Rendered by amb_sentinel.db.apply_schema() with:
--   {{VECTOR_DIM}}  <- database.postgresql.pgvector.dimension
--   {{IVF_LISTS}}   <- database.postgresql.pgvector.lists

CREATE TABLE IF NOT EXISTS agent_executions (
    id              SERIAL PRIMARY KEY,
    execution_id    TEXT UNIQUE NOT NULL,
    conversation_id TEXT NOT NULL,
    instruction     TEXT NOT NULL,
    status          VARCHAR(50) NOT NULL DEFAULT 'pending',
    result          JSONB,
    success         BOOLEAN DEFAULT FALSE,
    start_time      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time        TIMESTAMP,
    total_duration  FLOAT,
    iterations      INTEGER DEFAULT 0,
    tokens_used     INTEGER DEFAULT 0,
    steps_count     INTEGER DEFAULT 0,
    tools_used      JSONB DEFAULT '[]',
    error_message   TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_agent_executions_status
    ON agent_executions(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_executions_conversation
    ON agent_executions(conversation_id);

CREATE TABLE IF NOT EXISTS raw_data (
    id                  SERIAL PRIMARY KEY,
    source_type         VARCHAR(100) NOT NULL,
    source_url          TEXT,
    content_type        VARCHAR(100),
    content             TEXT NOT NULL,
    content_hash        VARCHAR(64) UNIQUE NOT NULL,
    metadata            JSONB DEFAULT '{}',
    collected_by_agent  TEXT,
    agent_execution_id  TEXT,
    collection_method   VARCHAR(100),
    file_path           TEXT,
    file_size           INTEGER,
    processed           BOOLEAN DEFAULT FALSE,
    analyzed            BOOLEAN DEFAULT FALSE,
    analysis_id         TEXT,
    priority            INTEGER DEFAULT 5,
    tags                JSONB DEFAULT '[]',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_raw_data_analyzed
    ON raw_data(analyzed, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_data_hash
    ON raw_data(content_hash);

-- analysis_results: matches data_analyzer.output_schema in config.yaml
CREATE TABLE IF NOT EXISTS analysis_results (
    analysis_id         TEXT PRIMARY KEY,
    raw_data_id         BIGINT,
    content_hash        VARCHAR(64),
    content_preview     TEXT,
    domain              VARCHAR(50),
    threat_level        VARCHAR(50),
    severity_score      FLOAT,
    confidence          FLOAT CHECK (confidence BETWEEN 0 AND 1),
    categories          JSONB,
    entities            JSONB,
    keywords            JSONB,
    reasoning           TEXT,
    action_required     BOOLEAN,
    urgency             VARCHAR(50),
    location            TEXT,
    affected_population TEXT,
    incident_count      INTEGER,
    source_credibility  FLOAT,
    warnings            JSONB,
    metadata            JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_analysis_results_level
    ON analysis_results(threat_level, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_results_hash
    ON analysis_results(content_hash);

CREATE TABLE IF NOT EXISTS tool_executions (
    id            SERIAL PRIMARY KEY,
    tool_name     VARCHAR(100) NOT NULL,
    execution_id  TEXT,
    arguments     JSONB NOT NULL,
    result        JSONB,
    success       BOOLEAN NOT NULL,
    duration      FLOAT NOT NULL,
    error_message TEXT,
    metadata      JSONB DEFAULT '{}',
    executed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tool_executions_tool
    ON tool_executions(tool_name, executed_at DESC);

-- embeddings: dim and lists come from config.yaml
CREATE TABLE IF NOT EXISTS embeddings (
    id           TEXT PRIMARY KEY,
    embedding    vector({{VECTOR_DIM}}),
    payload      JSONB,
    source       TEXT,
    content_hash TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
     WHERE tablename='embeddings' AND indexname='idx_embeddings_embedding'
  ) THEN
    BEGIN
      CREATE INDEX idx_embeddings_embedding
          ON embeddings USING ivfflat (embedding vector_cosine_ops)
          WITH (lists = {{IVF_LISTS}});
    EXCEPTION WHEN others THEN
      RAISE NOTICE 'ivfflat index skipped: %', SQLERRM;
    END;
  END IF;
END $$;


-- agent_events: one row per observable action of an agent run (LLM call,
-- parsed action, tool call, gate verdict, DB write). Written by
-- amb_sentinel.observability.
CREATE TABLE IF NOT EXISTS agent_events (
    id       BIGSERIAL PRIMARY KEY,
    run_id   TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    ts       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    kind     VARCHAR(40) NOT NULL,
    step     VARCHAR(20),
    payload  JSONB NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_agent_events_run ON agent_events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_agent_events_kind ON agent_events(kind, ts DESC);

-- One-glance answer to "did the last runs actually do anything?"
CREATE OR REPLACE VIEW v_agent_run_summary AS
SELECT e.run_id,
       MIN(e.ts)                                            AS started,
       MAX(e.ts)                                            AS finished,
       COUNT(*) FILTER (WHERE e.kind = 'llm_response')      AS llm_calls,
       COUNT(*) FILTER (WHERE e.kind = 'tool_result')       AS tool_calls,
       COUNT(*) FILTER (WHERE e.kind = 'tool_result' AND (e.payload->>'ok')::boolean) AS tool_ok,
       COUNT(*) FILTER (WHERE e.kind = 'gate_reject')       AS gate_rejects,
       (SELECT COUNT(*) FROM raw_data r
        WHERE r.agent_execution_id = e.run_id) AS raw_rows_written
FROM agent_events e
GROUP BY e.run_id;
