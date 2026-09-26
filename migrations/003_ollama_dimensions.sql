DROP INDEX IF EXISTS "sales-agent".idx_document_chunks_embedding;

ALTER TABLE "sales-agent".document_chunks
    ALTER COLUMN embedding TYPE vector(768)
    USING CASE WHEN embedding IS NULL THEN NULL ELSE NULL::vector(768) END;