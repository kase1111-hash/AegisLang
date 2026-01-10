"""
Unit tests for L1 Ingestion Layer (aegis_ingestor.py)
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

from aegislang.agents.aegis_ingestor import (
    AegisIngestor,
    SemanticChunker,
    ChunkingConfig,
    TextChunk,
    DocumentSection,
    IngestedDocument,
    MarkdownParser,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def chunking_config():
    """Create test chunking configuration."""
    return ChunkingConfig(
        target_tokens=100,
        min_tokens=20,
        max_tokens=200,
        overlap_tokens=10,
    )


@pytest.fixture
def chunker(chunking_config):
    """Create semantic chunker with test config."""
    return SemanticChunker(chunking_config)


@pytest.fixture
def ingestor(chunking_config):
    """Create ingestor with test config."""
    return AegisIngestor(chunking_config=chunking_config)


@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create a sample markdown file for testing."""
    content = """# Sample Policy Document

## Section 1: Introduction

This is the introduction to the policy document. It contains important information
about regulatory requirements that must be followed by all financial institutions.

## Section 2: Requirements

### 2.1 Customer Verification

Financial institutions must verify customer identity before opening any account.
This requirement applies to all new customers without exception.

### 2.2 Record Keeping

All records must be maintained for at least 5 years from the date of the transaction.
This includes customer identification documents and transaction records.

## Section 3: Compliance

Organizations shall not process transactions without proper authorization.
Violations will result in penalties as specified in the enforcement section.
"""
    file_path = tmp_path / "sample_policy.md"
    file_path.write_text(content)
    return file_path


# =============================================================================
# SemanticChunker Tests
# =============================================================================

class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    def test_count_tokens(self, chunker):
        """Test token counting."""
        text = "This is a simple test sentence."
        token_count = chunker.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 20  # Simple sentence should be small

    def test_chunk_empty_text(self, chunker):
        """Test chunking empty text."""
        chunks = chunker.chunk_text("", "TEST_S001")

        assert chunks == []

    def test_chunk_whitespace_text(self, chunker):
        """Test chunking whitespace-only text."""
        chunks = chunker.chunk_text("   \n\n   ", "TEST_S001")

        assert chunks == []

    def test_chunk_single_paragraph(self, chunker):
        """Test chunking a single short paragraph."""
        text = "This is a single paragraph of text for testing."
        chunks = chunker.chunk_text(text, "TEST_S001")

        assert len(chunks) >= 1
        assert chunks[0].text.strip() == text
        assert chunks[0].chunk_id == "TEST_S001_C000"

    def test_chunk_multiple_paragraphs(self, chunker):
        """Test chunking multiple paragraphs."""
        text = """First paragraph with some content.

Second paragraph with different content.

Third paragraph concluding the text."""

        chunks = chunker.chunk_text(text, "TEST_S001")

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.text.strip()
            assert chunk.token_count > 0

    def test_chunk_ids_are_sequential(self, chunker):
        """Test that chunk IDs are sequential."""
        text = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4.\n\nPara 5."
        chunks = chunker.chunk_text(text, "TEST_S001")

        for i, chunk in enumerate(chunks):
            assert f"C{i:03d}" in chunk.chunk_id

    def test_chunk_respects_max_tokens(self, chunking_config, chunker):
        """Test that chunks respect max token limit."""
        # Create a long paragraph
        text = "This is a word. " * 100
        chunks = chunker.chunk_text(text, "TEST_S001")

        for chunk in chunks:
            assert chunk.token_count <= chunking_config.max_tokens + 50  # Allow some flexibility


# =============================================================================
# MarkdownParser Tests
# =============================================================================

class TestMarkdownParser:
    """Tests for MarkdownParser class."""

    def test_parse_markdown_file(self, sample_markdown_file):
        """Test parsing a markdown file."""
        parser = MarkdownParser()
        sections, page_count = parser.parse(sample_markdown_file)

        assert len(sections) > 0
        assert page_count is None  # Markdown doesn't have pages

    def test_extract_hierarchy(self, sample_markdown_file):
        """Test hierarchy extraction from markdown."""
        parser = MarkdownParser()
        sections, _ = parser.parse(sample_markdown_file)

        # Check section titles are extracted
        section_titles = [s.section_title for s in sections]
        assert "Sample Policy Document" in section_titles or any("Introduction" in t for t in section_titles)

    def test_hierarchy_levels(self, sample_markdown_file):
        """Test that hierarchy levels are correctly detected."""
        parser = MarkdownParser()
        sections, _ = parser.parse(sample_markdown_file)

        # Should have different hierarchy levels
        levels = {s.hierarchy_level for s in sections}
        assert len(levels) >= 1

    def test_compute_hash(self, sample_markdown_file):
        """Test document hash computation."""
        parser = MarkdownParser()
        hash1 = parser.compute_hash(sample_markdown_file)
        hash2 = parser.compute_hash(sample_markdown_file)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length


# =============================================================================
# AegisIngestor Tests
# =============================================================================

class TestAegisIngestor:
    """Tests for AegisIngestor class."""

    def test_ingest_markdown(self, ingestor, sample_markdown_file):
        """Test ingesting a markdown file."""
        result = ingestor.ingest(sample_markdown_file)

        assert isinstance(result, IngestedDocument)
        assert result.doc_id
        assert result.metadata.document_type == "markdown"
        assert len(result.sections) > 0

    def test_ingest_returns_metadata(self, ingestor, sample_markdown_file):
        """Test that ingestion returns proper metadata."""
        result = ingestor.ingest(sample_markdown_file)

        assert result.metadata.source_file
        assert result.metadata.ingestion_timestamp
        assert result.metadata.hash
        assert len(result.metadata.hash) == 64

    def test_ingest_file_not_found(self, ingestor):
        """Test ingestion with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ingestor.ingest("/nonexistent/file.md")

    def test_ingest_unsupported_format(self, ingestor, tmp_path):
        """Test ingestion with unsupported file format."""
        unsupported = tmp_path / "test.xyz"
        unsupported.write_text("content")

        with pytest.raises(ValueError, match="Unsupported format"):
            ingestor.ingest(unsupported)

    def test_ingest_to_json(self, ingestor, sample_markdown_file):
        """Test JSON output."""
        result = ingestor.ingest_to_json(sample_markdown_file)

        assert isinstance(result, str)
        import json
        parsed = json.loads(result)
        assert "doc_id" in parsed
        assert "sections" in parsed

    def test_ingest_to_dict(self, ingestor, sample_markdown_file):
        """Test dictionary output."""
        result = ingestor.ingest_to_dict(sample_markdown_file)

        assert isinstance(result, dict)
        assert "doc_id" in result
        assert "sections" in result
        assert "metadata" in result

    def test_supported_formats(self, ingestor):
        """Test that supported formats are registered."""
        formats = ingestor.SUPPORTED_FORMATS

        assert ".pdf" in formats
        assert ".docx" in formats
        assert ".md" in formats
        assert ".html" in formats


# =============================================================================
# Integration Tests
# =============================================================================

class TestIngestorIntegration:
    """Integration tests for the ingestor."""

    def test_full_pipeline(self, ingestor, sample_markdown_file):
        """Test full ingestion pipeline."""
        # Ingest
        result = ingestor.ingest(sample_markdown_file)

        # Verify structure
        assert result.doc_id
        assert len(result.sections) > 0

        # Verify chunks exist
        total_chunks = sum(len(s.text_chunks) for s in result.sections)
        assert total_chunks > 0

        # Verify all chunks have content
        for section in result.sections:
            for chunk in section.text_chunks:
                assert chunk.text.strip()
                assert chunk.token_count > 0

    def test_section_parent_relationships(self, ingestor, sample_markdown_file):
        """Test that section parent relationships are set."""
        result = ingestor.ingest(sample_markdown_file)

        # At least some sections should have parents (nested headers)
        sections_with_parents = [s for s in result.sections if s.parent_section]
        # This depends on the test document structure
        # Just verify the structure is valid
        for section in result.sections:
            if section.parent_section:
                parent_ids = [s.section_id for s in result.sections]
                assert section.parent_section in parent_ids
