DROP INDEX IF EXISTS "sales-agent".idx_document_chunks_embedding;

ALTER TABLE "sales-agent".document_chunks
    ALTER COLUMN embedding TYPE vector(2048)
    USING CASE WHEN embedding IS NULL THEN NULL ELSE NULL::vector(2048) END;

CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_tenant_source_url
    ON "sales-agent".documents (tenant_id, source_url);

