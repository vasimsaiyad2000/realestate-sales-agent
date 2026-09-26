CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS "sales-agent";

ALTER TABLE IF EXISTS "sales-agent".tenants
    ADD COLUMN IF NOT EXISTS timezone text NOT NULL DEFAULT 'UTC',
    ADD COLUMN IF NOT EXISTS active boolean NOT NULL DEFAULT true;

CREATE TABLE IF NOT EXISTS "sales-agent".conversations (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id bigint NOT NULL REFERENCES "sales-agent".tenants(id) ON DELETE CASCADE,
    customer_phone text NOT NULL,
    state text NOT NULL DEFAULT 'qualifying',
    opted_out boolean NOT NULL DEFAULT false,
    last_message_at timestamptz,
    UNIQUE (tenant_id, customer_phone)
);

CREATE TABLE IF NOT EXISTS "sales-agent".messages (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id bigint NOT NULL REFERENCES "sales-agent".tenants(id) ON DELETE CASCADE,
    conversation_id bigint NOT NULL REFERENCES "sales-agent".conversations(id) ON DELETE CASCADE,
    provider_message_id text NOT NULL,
    direction text NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    message_type text NOT NULL,
    body text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, provider_message_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
    ON "sales-agent".messages (conversation_id, created_at);

CREATE TABLE IF NOT EXISTS "sales-agent".documents (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id bigint NOT NULL REFERENCES "sales-agent".tenants(id) ON DELETE CASCADE,
    project_id bigint REFERENCES "sales-agent".projects(id) ON DELETE CASCADE,
    name text NOT NULL,
    source_url text NOT NULL,
    checksum text NOT NULL,
    status text NOT NULL DEFAULT 'pending',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, checksum),
    UNIQUE (tenant_id, source_url)
);

CREATE TABLE IF NOT EXISTS "sales-agent".document_chunks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id bigint NOT NULL REFERENCES "sales-agent".tenants(id) ON DELETE CASCADE,
    document_id bigint NOT NULL REFERENCES "sales-agent".documents(id) ON DELETE CASCADE,
    project_id bigint REFERENCES "sales-agent".projects(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    page_number integer,
    section text,
    embedding vector(2048),
    search_text tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_document_chunks_tenant_project
    ON "sales-agent".document_chunks (tenant_id, project_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_search_text
    ON "sales-agent".document_chunks USING gin (search_text);
