"""
AegisLang L2 Parsing Layer - Policy Parser Agent

Purpose: Identify clause structure and extract semantic components from policy language.

Functional Requirements:
- PRS-001: Detect clause type (obligation, prohibition, conditional, definition, exception)
- PRS-002: Extract actor entity from clause
- PRS-003: Extract action/verb phrase from clause
- PRS-004: Extract object/target entity from clause
- PRS-005: Extract conditional triggers
- PRS-006: Extract temporal scope (deadlines, frequencies)
- PRS-007: Represent parsed clauses as semantic triples
- PRS-008: Handle cross-references between clauses
- PRS-009: Confidence scoring for each extraction
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
# Schema Definitions
# -----------------------------------------------------------------------------


class ClauseType(str, Enum):
    """Types of regulatory clauses."""

    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    CONDITIONAL = "conditional"
    DEFINITION = "definition"
    EXCEPTION = "exception"


class ActorEntity(BaseModel):
    """Entity responsible for an action."""

    entity: str = Field(..., description="The actor entity name")
    qualifiers: list[str] = Field(
        default_factory=list, description="Qualifiers or modifiers for the entity"
    )


class ActionPhrase(BaseModel):
    """Action or verb phrase from a clause."""

    verb: str = Field(..., description="The main verb or action")
    modifiers: list[str] = Field(
        default_factory=list, description="Adverbs or modifiers for the action"
    )


class ObjectEntity(BaseModel):
    """Target entity of an action."""

    entity: str = Field(..., description="The object entity name")
    qualifiers: list[str] = Field(
        default_factory=list, description="Qualifiers or modifiers for the entity"
    )


class Condition(BaseModel):
    """Conditional trigger for a clause."""

    trigger: str = Field(..., description="The triggering condition")
    temporal: str | None = Field(
        default=None, description="Temporal aspect of the condition"
    )


class TemporalScope(BaseModel):
    """Temporal scope of a clause."""

    deadline: str | None = Field(default=None, description="Deadline if specified")
    frequency: str | None = Field(default=None, description="Frequency if specified")
    duration: str | None = Field(default=None, description="Duration if specified")


class ParsedClause(BaseModel):
    """Output schema for a parsed clause."""

    clause_id: str = Field(
        ..., description="Unique clause identifier (doc_id + section + sequence)"
    )
    source_chunk_id: str = Field(
        ..., description="Reference to originating text chunk"
    )
    source_text: str = Field(
        ..., description="Original clause text for traceability"
    )
    type: ClauseType = Field(..., description="Type of clause")
    actor: ActorEntity = Field(..., description="Entity responsible for the action")
    action: ActionPhrase = Field(..., description="The action or verb phrase")
    object: ObjectEntity | None = Field(
        default=None, description="Target of the action"
    )
    condition: Condition | None = Field(
        default=None, description="Triggering condition"
    )
    temporal_scope: TemporalScope | None = Field(
        default=None, description="Temporal constraints"
    )
    cross_references: list[str] = Field(
        default_factory=list, description="References to other clause_ids"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for extraction"
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Round confidence to 2 decimal places."""
        return round(v, 2)


class ParsedClauseCollection(BaseModel):
    """Collection of parsed clauses from a document."""

    doc_id: str = Field(..., description="Source document ID")
    clauses: list[ParsedClause] = Field(
        default_factory=list, description="Parsed clauses"
    )
    parse_timestamp: str = Field(..., description="ISO 8601 timestamp of parsing")


# -----------------------------------------------------------------------------
# Prompt Templates
# -----------------------------------------------------------------------------

CLAUSE_PARSER_SYSTEM_PROMPT = """You are a legal language parser specializing in regulatory and policy documents.

Your task is to analyze regulatory text and extract structured semantic components. You must identify:

1. **Clause Type**: Determine if the clause is an obligation, prohibition, permission, conditional, definition, or exception.
   - obligation: Required action (must, shall, is required to)
   - prohibition: Forbidden action (must not, shall not, is prohibited from)
   - permission: Allowed action (may, is permitted to, can)
   - conditional: Action contingent on condition (if, when, where, unless)
   - definition: Term definition (means, refers to, is defined as)
   - exception: Carve-out from rule (except, unless, notwithstanding)

2. **Actor**: The entity responsible for the action (with any qualifiers)

3. **Action**: The verb phrase describing what must/must not/may be done

4. **Object**: The target of the action (if present)

5. **Condition**: Any triggering condition or prerequisite

6. **Temporal Scope**: Deadlines, frequencies, or durations mentioned

7. **Cross-References**: References to other sections, articles, or clauses

Always return valid JSON matching the specified schema. Be precise and extract only what is explicitly stated."""

CLAUSE_PARSER_USER_PROMPT = """Analyze the following clause and extract its semantic structure:

<clause>
{clause_text}
</clause>

Extract and return as JSON with the following structure:
{{
  "type": "obligation|prohibition|permission|conditional|definition|exception",
  "actor": {{
    "entity": "the responsible entity",
    "qualifiers": ["list", "of", "qualifiers"]
  }},
  "action": {{
    "verb": "the main action verb",
    "modifiers": ["list", "of", "modifiers"]
  }},
  "object": {{
    "entity": "target of the action",
    "qualifiers": ["list", "of", "qualifiers"]
  }} or null if not present,
  "condition": {{
    "trigger": "the triggering condition",
    "temporal": "temporal aspect if any"
  }} or null if not present,
  "temporal_scope": {{
    "deadline": "deadline if specified",
    "frequency": "frequency if specified",
    "duration": "duration if specified"
  }} or null if not present,
  "cross_references": ["list of referenced sections/articles"],
  "confidence": 0.0 to 1.0
}}

Return ONLY valid JSON, no additional text."""


# -----------------------------------------------------------------------------
# LLM Clients
# -----------------------------------------------------------------------------


class BaseLLMClient:
    """Base class for LLM clients."""

    def parse_clause(self, clause_text: str) -> dict[str, Any]:
        """Parse a clause and return structured data."""
        raise NotImplementedError


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client for clause parsing."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            ) from e

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def parse_clause(self, clause_text: str) -> dict[str, Any]:
        """Parse a clause using Claude."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=CLAUSE_PARSER_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": CLAUSE_PARSER_USER_PROMPT.format(clause_text=clause_text),
                }
            ],
        )

        response_text = message.content[0].text
        return self._extract_json(response_text)

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON from response text."""
        # Try to parse directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON from response: {text[:200]}")


class OpenAIClient(BaseLLMClient):
    """OpenAI client for clause parsing."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def parse_clause(self, clause_text: str) -> dict[str, Any]:
        """Parse a clause using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": CLAUSE_PARSER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CLAUSE_PARSER_USER_PROMPT.format(clause_text=clause_text),
                },
            ],
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content
        return json.loads(response_text)


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without API calls."""

    def __init__(self):
        # Ordered from most specific to least specific for proper matching
        self._clause_patterns = [
            ("must not", ClauseType.PROHIBITION),
            ("shall not", ClauseType.PROHIBITION),
            ("prohibited", ClauseType.PROHIBITION),
            ("defined as", ClauseType.DEFINITION),
            ("means", ClauseType.DEFINITION),
            ("except", ClauseType.EXCEPTION),
            ("unless", ClauseType.EXCEPTION),
            ("may", ClauseType.PERMISSION),
            ("permitted", ClauseType.PERMISSION),
            ("must", ClauseType.OBLIGATION),
            ("shall", ClauseType.OBLIGATION),
            ("required", ClauseType.OBLIGATION),
            ("if", ClauseType.CONDITIONAL),
            ("when", ClauseType.CONDITIONAL),
        ]

    def parse_clause(self, clause_text: str) -> dict[str, Any]:
        """Parse clause using pattern matching (for testing)."""
        clause_lower = clause_text.lower().strip()

        # Detect clause type
        clause_type = ClauseType.OBLIGATION  # default

        # Check for sentence-initial conditionals first
        if clause_lower.startswith("if ") or clause_lower.startswith("when "):
            clause_type = ClauseType.CONDITIONAL
        else:
            # Check patterns in priority order
            for pattern, ctype in self._clause_patterns:
                if pattern in clause_lower:
                    clause_type = ctype
                    break

        # Extract basic components using simple heuristics
        words = clause_text.split()

        # Find actor (typically first noun phrase before modal verb)
        actor = self._extract_actor(clause_text)

        # Find action verb
        action = self._extract_action(clause_text)

        # Find object
        obj = self._extract_object(clause_text)

        # Check for conditions
        condition = self._extract_condition(clause_text)

        # Check for temporal scope
        temporal = self._extract_temporal(clause_text)

        return {
            "type": clause_type.value,
            "actor": {"entity": actor, "qualifiers": []},
            "action": {"verb": action, "modifiers": []},
            "object": {"entity": obj, "qualifiers": []} if obj else None,
            "condition": condition,
            "temporal_scope": temporal,
            "cross_references": [],
            "confidence": 0.75,
        }

    def _extract_actor(self, text: str) -> str:
        """Extract actor from text."""
        # Simple pattern: words before modal verbs
        modal_pattern = r"^([\w\s]+?)\s+(?:must|shall|may|should|can)\s"
        match = re.search(modal_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "unspecified entity"

    def _extract_action(self, text: str) -> str:
        """Extract action verb from text."""
        # Pattern: word after modal verb
        modal_pattern = r"(?:must|shall|may|should|can)\s+(?:not\s+)?(\w+)"
        match = re.search(modal_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return "comply"

    def _extract_object(self, text: str) -> str | None:
        """Extract object from text."""
        # Simple pattern: words after action verb
        patterns = [
            r"(?:verify|maintain|report|submit|provide|ensure)\s+([\w\s]+?)(?:\.|,|$)",
            r"(?:must|shall)\s+\w+\s+([\w\s]+?)(?:\.|,|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                obj = match.group(1).strip()
                # Limit length
                words = obj.split()[:5]
                return " ".join(words)
        return None

    def _extract_condition(self, text: str) -> dict[str, str] | None:
        """Extract condition from text."""
        patterns = [
            r"(?:if|when|where)\s+([\w\s]+?)(?:,|then|$)",
            r"(?:before|after|upon)\s+([\w\s]+?)(?:,|\.|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return {"trigger": match.group(1).strip(), "temporal": None}
        return None

    def _extract_temporal(self, text: str) -> dict[str, str | None] | None:
        """Extract temporal scope from text."""
        result: dict[str, str | None] = {
            "deadline": None,
            "frequency": None,
            "duration": None,
        }

        # Deadline patterns
        deadline_match = re.search(
            r"(?:within|by|before)\s+([\w\s]+?)(?:\.|,|$)", text, re.IGNORECASE
        )
        if deadline_match:
            result["deadline"] = deadline_match.group(1).strip()

        # Duration patterns
        duration_match = re.search(
            r"(?:for|at least|up to)\s+(\d+\s*(?:days?|months?|years?))",
            text,
            re.IGNORECASE,
        )
        if duration_match:
            result["duration"] = duration_match.group(1).strip()

        # Frequency patterns
        frequency_match = re.search(
            r"(?:annually|monthly|weekly|daily|quarterly)", text, re.IGNORECASE
        )
        if frequency_match:
            result["frequency"] = frequency_match.group(0)

        if any(result.values()):
            return result
        return None


# -----------------------------------------------------------------------------
# Policy Parser Agent
# -----------------------------------------------------------------------------


class PolicyParserAgent:
    """
    L2 Parsing Layer Agent.

    Extracts semantic structure from policy clauses using LLM-based parsing.
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        use_mock: bool = False,
    ):
        """
        Initialize the parser agent.

        Args:
            llm_provider: LLM provider ("anthropic" or "openai")
            api_key: API key (or set via environment variable)
            model: Model identifier (uses provider default if not specified)
            use_mock: Use mock client for testing
        """
        if use_mock:
            self.llm_client: BaseLLMClient = MockLLMClient()
            logger.info("policy_parser_initialized", provider="mock")
        elif llm_provider == "anthropic":
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if model:
                kwargs["model"] = model
            self.llm_client = AnthropicClient(**kwargs)
            logger.info("policy_parser_initialized", provider="anthropic", model=model)
        elif llm_provider == "openai":
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if model:
                kwargs["model"] = model
            self.llm_client = OpenAIClient(**kwargs)
            logger.info("policy_parser_initialized", provider="openai", model=model)
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")

    def parse_clause(
        self,
        clause_text: str,
        clause_id: str,
        source_chunk_id: str,
    ) -> ParsedClause:
        """
        Parse a single clause and extract semantic structure.

        Args:
            clause_text: The raw clause text
            clause_id: Unique identifier for this clause
            source_chunk_id: Reference to source chunk

        Returns:
            ParsedClause with extracted structure
        """
        logger.debug(
            "parsing_clause",
            clause_id=clause_id,
            text_length=len(clause_text),
        )

        try:
            # Get LLM response
            parsed_data = self.llm_client.parse_clause(clause_text)

            # Build ParsedClause from response
            clause = ParsedClause(
                clause_id=clause_id,
                source_chunk_id=source_chunk_id,
                source_text=clause_text,
                type=ClauseType(parsed_data["type"]),
                actor=ActorEntity(**parsed_data["actor"]),
                action=ActionPhrase(**parsed_data["action"]),
                object=ObjectEntity(**parsed_data["object"])
                if parsed_data.get("object")
                else None,
                condition=Condition(**parsed_data["condition"])
                if parsed_data.get("condition")
                else None,
                temporal_scope=TemporalScope(**parsed_data["temporal_scope"])
                if parsed_data.get("temporal_scope")
                else None,
                cross_references=parsed_data.get("cross_references", []),
                confidence=parsed_data.get("confidence", 0.5),
            )

            logger.info(
                "clause_parsed",
                clause_id=clause_id,
                type=clause.type.value,
                confidence=clause.confidence,
            )

            return clause

        except Exception as e:
            logger.error(
                "clause_parse_failed",
                clause_id=clause_id,
                error=str(e),
            )
            raise

    def parse_text_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        doc_id: str,
    ) -> list[ParsedClause]:
        """
        Parse a text chunk and extract all clauses.

        Splits text into individual clauses and parses each.

        Args:
            chunk_text: The text chunk to parse
            chunk_id: Source chunk identifier
            doc_id: Parent document identifier

        Returns:
            List of ParsedClause objects
        """
        # Split into sentences/clauses
        clauses_text = self._split_into_clauses(chunk_text)

        parsed_clauses: list[ParsedClause] = []

        for i, clause_text in enumerate(clauses_text):
            clause_id = f"{doc_id}_{chunk_id}_CL{i + 1:03d}"

            try:
                parsed = self.parse_clause(
                    clause_text=clause_text,
                    clause_id=clause_id,
                    source_chunk_id=chunk_id,
                )
                parsed_clauses.append(parsed)
            except Exception as e:
                logger.warning(
                    "skipping_unparseable_clause",
                    clause_id=clause_id,
                    error=str(e),
                )
                continue

        return parsed_clauses

    def parse_ingested_document(
        self,
        ingested_doc: dict[str, Any],
    ) -> ParsedClauseCollection:
        """
        Parse all clauses from an ingested document.

        Args:
            ingested_doc: IngestedDocument as dictionary

        Returns:
            ParsedClauseCollection with all parsed clauses
        """
        from datetime import datetime, timezone

        doc_id = ingested_doc["doc_id"]
        all_clauses: list[ParsedClause] = []

        for section in ingested_doc.get("sections", []):
            section_id = section["section_id"]

            for chunk in section.get("text_chunks", []):
                chunk_id = chunk["chunk_id"]
                chunk_text = chunk["text"]

                # Parse clauses from chunk
                clauses = self.parse_text_chunk(
                    chunk_text=chunk_text,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                )
                all_clauses.extend(clauses)

        collection = ParsedClauseCollection(
            doc_id=doc_id,
            clauses=all_clauses,
            parse_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "document_parsed",
            doc_id=doc_id,
            total_clauses=len(all_clauses),
        )

        return collection

    def _split_into_clauses(self, text: str) -> list[str]:
        """
        Split text into individual clauses.

        Uses sentence boundaries and regulatory markers.
        """
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)

        clauses: list[str] = []
        current_clause: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if this is a regulatory clause (contains modal verbs or key terms)
            is_regulatory = bool(
                re.search(
                    r"\b(must|shall|may|should|required|prohibited|permitted|"
                    r"means|defined|except|unless|if|when|where)\b",
                    sentence,
                    re.IGNORECASE,
                )
            )

            if is_regulatory:
                # Flush any accumulated context
                if current_clause:
                    # Add context as prefix to this clause if short
                    context = " ".join(current_clause)
                    if len(context) < 100:
                        sentence = f"{context} {sentence}"
                    current_clause = []

                clauses.append(sentence)
            else:
                # Accumulate as context
                current_clause.append(sentence)

        # Handle remaining context
        if current_clause and not clauses:
            # No regulatory clauses found, treat entire text as one clause
            clauses.append(" ".join(current_clause))

        return clauses


# -----------------------------------------------------------------------------
# Event Publishing (Agent-OS Integration)
# -----------------------------------------------------------------------------


async def publish_parsed_event(
    collection: ParsedClauseCollection,
    redis_url: str | None = None,
) -> None:
    """
    Publish policy.parsed event to Agent-OS event bus.

    Args:
        collection: The parsed clause collection
        redis_url: Redis connection URL
    """
    if redis_url is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    try:
        import redis.asyncio as redis_async

        client = redis_async.from_url(redis_url)
        await client.publish(
            "policy.parsed",
            collection.model_dump_json(),
        )
        await client.aclose()

        logger.info(
            "event_published",
            topic="policy.parsed",
            doc_id=collection.doc_id,
            clause_count=len(collection.clauses),
        )
    except Exception as e:
        logger.warning(
            "event_publish_failed",
            topic="policy.parsed",
            error=str(e),
        )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for policy parsing."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AegisLang Policy Parser - L2 Parsing Layer"
    )
    parser.add_argument(
        "input",
        help="Input file (JSON from L1 ingestor) or raw text file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "mock"],
        default="anthropic",
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        help="Model identifier (uses provider default if not specified)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Treat input as raw text instead of JSON",
    )

    args = parser.parse_args()

    # Initialize parser
    use_mock = args.provider == "mock"
    provider = "anthropic" if use_mock else args.provider

    try:
        agent = PolicyParserAgent(
            llm_provider=provider,
            model=args.model,
            use_mock=use_mock,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Read input
    from pathlib import Path

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    input_text = input_path.read_text(encoding="utf-8")

    try:
        if args.raw:
            # Parse raw text
            from datetime import datetime, timezone

            clauses = agent.parse_text_chunk(
                chunk_text=input_text,
                chunk_id="CLI_INPUT",
                doc_id="CLI_DOC",
            )
            result = ParsedClauseCollection(
                doc_id="CLI_DOC",
                clauses=clauses,
                parse_timestamp=datetime.now(timezone.utc).isoformat(),
            )
        else:
            # Parse JSON from L1 ingestor
            ingested_doc = json.loads(input_text)
            result = agent.parse_ingested_document(ingested_doc)

        output_json = result.model_dump_json(indent=2)

        if args.output:
            Path(args.output).write_text(output_json)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output_json)

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
