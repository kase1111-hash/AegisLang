"""
AegisLang L4 Compilation Layer - Compiler Agent

Purpose: Translate parsed and mapped clauses into executable artifacts
across multiple target formats.

Functional Requirements:
- CMP-001: Generate YAML compliance rule definitions
- CMP-002: Generate SQL check constraints
- CMP-003: Generate Python test stubs
- CMP-004: Generate Terraform policy configs
- CMP-005: Generate OPA/Rego policies
- CMP-006: Support pluggable Jinja2 template system
- CMP-007: Validate generated artifacts syntactically
- CMP-008: Embed source clause references in artifacts
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, BaseLoader, TemplateNotFound
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

VERSION = "1.0.0"


# -----------------------------------------------------------------------------
# Schema Definitions
# -----------------------------------------------------------------------------


class ArtifactFormat(str, Enum):
    """Supported output formats."""

    YAML = "yaml"
    SQL = "sql"
    PYTHON = "python"
    TERRAFORM = "terraform"
    REGO = "rego"
    JSON = "json"


class CompiledArtifact(BaseModel):
    """Output schema for a compiled artifact."""

    artifact_id: str = Field(..., description="Unique artifact identifier")
    clause_id: str = Field(..., description="Source clause ID")
    format: ArtifactFormat = Field(..., description="Output format")
    content: str = Field(..., description="Generated artifact content")
    file_path: str = Field(..., description="Suggested file path")
    syntax_valid: bool = Field(..., description="Whether syntax is valid")
    template_used: str = Field(..., description="Template name used")
    compilation_timestamp: str = Field(..., description="ISO 8601 timestamp")
    warnings: list[str] = Field(
        default_factory=list, description="Compilation warnings"
    )


class CompiledArtifactCollection(BaseModel):
    """Collection of compiled artifacts."""

    doc_id: str = Field(..., description="Source document ID")
    artifacts: list[CompiledArtifact] = Field(
        default_factory=list, description="Compiled artifacts"
    )
    compilation_timestamp: str = Field(..., description="ISO 8601 timestamp")
    formats_generated: list[str] = Field(
        default_factory=list, description="Formats generated"
    )


# -----------------------------------------------------------------------------
# Built-in Templates
# -----------------------------------------------------------------------------

YAML_OBLIGATION_TEMPLATE = '''# Source: {{ clause.source_text | truncate(80) }}
# Clause ID: {{ clause.clause_id }}
# Generated: {{ timestamp }}
# Confidence: {{ confidence }}

control:
  id: {{ clause.clause_id }}
  type: {{ clause.type }}
  rule: "{{ clause.action.verb }}{% if clause.object %} {{ clause.object.entity }}{% endif %}"

  actor:
    entity: "{{ clause.actor.entity }}"
{% if mappings.actor_path %}
    mapped_to: "{{ mappings.actor_path }}"
{% endif %}

  action:
    operation: "{{ clause.action.verb }}"
{% if clause.action.modifiers %}
    modifiers:
{% for mod in clause.action.modifiers %}
      - "{{ mod }}"
{% endfor %}
{% endif %}

{% if clause.object %}
  object:
    entity: "{{ clause.object.entity }}"
{% if mappings.object_path %}
    mapped_to: "{{ mappings.object_path }}"
{% endif %}
{% endif %}

{% if clause.condition %}
  trigger:
    event: "{{ clause.condition.trigger }}"
{% if clause.condition.temporal %}
    temporal: "{{ clause.condition.temporal }}"
{% endif %}
{% endif %}

{% if clause.temporal_scope %}
  temporal_scope:
{% if clause.temporal_scope.deadline %}
    deadline: "{{ clause.temporal_scope.deadline }}"
{% endif %}
{% if clause.temporal_scope.frequency %}
    frequency: "{{ clause.temporal_scope.frequency }}"
{% endif %}
{% if clause.temporal_scope.duration %}
    duration: "{{ clause.temporal_scope.duration }}"
{% endif %}
{% endif %}

  severity: {{ severity }}

  metadata:
    source_document: "{{ doc_id }}"
    confidence: {{ confidence }}
    generated_by: "aegislang-compiler-v{{ version }}"
'''

YAML_PROHIBITION_TEMPLATE = '''# Source: {{ clause.source_text | truncate(80) }}
# Clause ID: {{ clause.clause_id }}
# Generated: {{ timestamp }}
# Confidence: {{ confidence }}

control:
  id: {{ clause.clause_id }}
  type: prohibition
  rule: "MUST NOT {{ clause.action.verb }}{% if clause.object %} {{ clause.object.entity }}{% endif %}"

  actor:
    entity: "{{ clause.actor.entity }}"
{% if mappings.actor_path %}
    mapped_to: "{{ mappings.actor_path }}"
{% endif %}

  prohibited_action:
    operation: "{{ clause.action.verb }}"
    enforcement: "block"

{% if clause.object %}
  object:
    entity: "{{ clause.object.entity }}"
{% if mappings.object_path %}
    mapped_to: "{{ mappings.object_path }}"
{% endif %}
{% endif %}

{% if clause.condition %}
  exception:
    condition: "{{ clause.condition.trigger }}"
{% endif %}

  severity: critical

  metadata:
    source_document: "{{ doc_id }}"
    confidence: {{ confidence }}
    generated_by: "aegislang-compiler-v{{ version }}"
'''

SQL_CHECK_CONSTRAINT_TEMPLATE = '''-- Source: {{ clause.source_text | truncate(70) }}
-- Clause ID: {{ clause.clause_id }}
-- Generated: {{ timestamp }}
-- Confidence: {{ confidence }}

{% if clause.type == 'obligation' %}
-- Obligation: {{ clause.actor.entity }} must {{ clause.action.verb }}
ALTER TABLE {{ table_name }}
ADD CONSTRAINT chk_{{ clause.clause_id | lower | replace('-', '_') }}
CHECK (
    {{ check_condition }}
);

-- Trigger for enforcement
CREATE OR REPLACE FUNCTION enforce_{{ clause.clause_id | lower | replace('-', '_') }}()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT ({{ check_condition }}) THEN
        RAISE EXCEPTION 'Compliance violation: {{ clause.clause_id }} - {{ clause.actor.entity }} must {{ clause.action.verb }}';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_{{ clause.clause_id | lower | replace('-', '_') }}
BEFORE INSERT OR UPDATE ON {{ table_name }}
FOR EACH ROW
EXECUTE FUNCTION enforce_{{ clause.clause_id | lower | replace('-', '_') }}();

{% elif clause.type == 'prohibition' %}
-- Prohibition: {{ clause.actor.entity }} must not {{ clause.action.verb }}
ALTER TABLE {{ table_name }}
ADD CONSTRAINT chk_{{ clause.clause_id | lower | replace('-', '_') }}_prohibit
CHECK (
    NOT ({{ check_condition }})
);

{% endif %}

COMMENT ON CONSTRAINT chk_{{ clause.clause_id | lower | replace('-', '_') }}{% if clause.type == 'prohibition' %}_prohibit{% endif %} ON {{ table_name }}
IS 'AegisLang: {{ clause.source_text | truncate(200) }}';
'''

PYTHON_TEST_TEMPLATE = '''"""
Test for compliance rule: {{ clause.clause_id }}

Source: {{ clause.source_text | truncate(70) }}
Generated: {{ timestamp }}
Confidence: {{ confidence }}
"""

import pytest
from datetime import datetime, timedelta


class Test{{ clause.clause_id | replace('-', '_') | replace('.', '_') }}:
    """Test cases for {{ clause.type }} rule: {{ clause.clause_id }}"""

    @pytest.fixture
    def compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}(self):
        """Create a compliant {{ clause.actor.entity }} fixture."""
        return {
            "id": "test_{{ clause.actor.entity | lower | replace(' ', '_') }}_001",
{% if clause.type == 'obligation' %}
            "{{ clause.action.verb | lower }}_status": True,
{% elif clause.type == 'prohibition' %}
            "{{ clause.action.verb | lower }}_status": False,
{% endif %}
{% if clause.object %}
            "{{ clause.object.entity | lower | replace(' ', '_') }}": "valid_value",
{% endif %}
            "created_at": datetime.utcnow(),
        }

    @pytest.fixture
    def non_compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}(self):
        """Create a non-compliant {{ clause.actor.entity }} fixture."""
        return {
            "id": "test_{{ clause.actor.entity | lower | replace(' ', '_') }}_002",
{% if clause.type == 'obligation' %}
            "{{ clause.action.verb | lower }}_status": False,
{% elif clause.type == 'prohibition' %}
            "{{ clause.action.verb | lower }}_status": True,
{% endif %}
{% if clause.object %}
            "{{ clause.object.entity | lower | replace(' ', '_') }}": None,
{% endif %}
            "created_at": datetime.utcnow(),
        }

{% if clause.type == 'obligation' %}
    def test_{{ clause.action.verb | lower }}_obligation_met(
        self, compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}
    ):
        """Test that obligation is satisfied when {{ clause.action.verb }} is performed."""
        entity = compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}

        # Assert obligation is met
        assert entity["{{ clause.action.verb | lower }}_status"] is True, \\
            "{{ clause.actor.entity }} must {{ clause.action.verb }}{% if clause.object %} {{ clause.object.entity }}{% endif %}"

    def test_{{ clause.action.verb | lower }}_obligation_violated(
        self, non_compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}
    ):
        """Test that violation is detected when {{ clause.action.verb }} is not performed."""
        entity = non_compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}

        # Assert obligation is violated
        assert entity["{{ clause.action.verb | lower }}_status"] is False, \\
            "Expected violation when {{ clause.actor.entity }} does not {{ clause.action.verb }}"

{% elif clause.type == 'prohibition' %}
    def test_{{ clause.action.verb | lower }}_prohibition_respected(
        self, compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}
    ):
        """Test that prohibition is respected when {{ clause.action.verb }} is not performed."""
        entity = compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}

        # Assert prohibition is respected
        assert entity["{{ clause.action.verb | lower }}_status"] is False, \\
            "{{ clause.actor.entity }} must not {{ clause.action.verb }}{% if clause.object %} {{ clause.object.entity }}{% endif %}"

    def test_{{ clause.action.verb | lower }}_prohibition_violated(
        self, non_compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}
    ):
        """Test that violation is detected when {{ clause.action.verb }} is performed."""
        entity = non_compliant_{{ clause.actor.entity | lower | replace(' ', '_') }}

        # Assert prohibition is violated
        assert entity["{{ clause.action.verb | lower }}_status"] is True, \\
            "Expected violation when {{ clause.actor.entity }} does {{ clause.action.verb }}"
{% endif %}

{% if clause.condition %}
    def test_condition_trigger(self):
        """Test that rule applies when condition '{{ clause.condition.trigger }}' is met."""
        # TODO: Implement condition testing for: {{ clause.condition.trigger }}
        pytest.skip("Condition testing not yet implemented")
{% endif %}

{% if clause.temporal_scope and clause.temporal_scope.deadline %}
    def test_deadline_compliance(self):
        """Test deadline compliance: {{ clause.temporal_scope.deadline }}"""
        # TODO: Implement deadline testing
        pytest.skip("Deadline testing not yet implemented")
{% endif %}


# Metadata for test discovery
CLAUSE_ID = "{{ clause.clause_id }}"
CLAUSE_TYPE = "{{ clause.type }}"
SOURCE_TEXT = """{{ clause.source_text }}"""
CONFIDENCE = {{ confidence }}
'''

TERRAFORM_POLICY_TEMPLATE = '''# Source: {{ clause.source_text | truncate(70) }}
# Clause ID: {{ clause.clause_id }}
# Generated: {{ timestamp }}
# Confidence: {{ confidence }}

# Sentinel Policy for {{ clause.type }}: {{ clause.clause_id }}

resource "sentinel_policy" "{{ clause.clause_id | lower | replace('-', '_') | replace('.', '_') }}" {
  name        = "{{ clause.clause_id }}"
  description = "{{ clause.source_text | truncate(200) }}"

  enforcement_level = {% if clause.type == 'prohibition' %}"hard-mandatory"{% else %}"soft-mandatory"{% endif %}


  policy = <<-POLICY
    import "tfplan/v2" as tfplan

    # {{ clause.type | title }}: {{ clause.actor.entity }} {{ 'must' if clause.type == 'obligation' else 'must not' }} {{ clause.action.verb }}

    main = rule {
{% if clause.type == 'obligation' %}
      all tfplan.resource_changes as _, rc {
        rc.type is "{{ resource_type }}" and
        rc.change.after.{{ clause.action.verb | lower }}_enabled is true
      }
{% elif clause.type == 'prohibition' %}
      all tfplan.resource_changes as _, rc {
        rc.type is "{{ resource_type }}" implies
        rc.change.after.{{ clause.action.verb | lower }}_enabled is not true
      }
{% else %}
      true  # Permissive rule for {{ clause.type }}
{% endif %}
    }
  POLICY

  tags = {
    aegislang_clause_id = "{{ clause.clause_id }}"
    aegislang_type      = "{{ clause.type }}"
    aegislang_version   = "{{ version }}"
    confidence          = "{{ confidence }}"
  }
}
'''

REGO_POLICY_TEMPLATE = '''# Source: {{ clause.source_text | truncate(70) }}
# Clause ID: {{ clause.clause_id }}
# Generated: {{ timestamp }}
# Confidence: {{ confidence }}

package aegislang.{{ clause.clause_id | lower | replace('-', '_') | replace('.', '_') }}

import future.keywords.if
import future.keywords.in

# {{ clause.type | title }}: {{ clause.actor.entity }} {{ 'must' if clause.type == 'obligation' else 'must not' }} {{ clause.action.verb }}

default allow := false

{% if clause.type == 'obligation' %}
# Obligation rule: {{ clause.action.verb }} is required
allow if {
    input.actor == "{{ clause.actor.entity | lower }}"
    input.action == "{{ clause.action.verb | lower }}"
{% if clause.object %}
    input.object == "{{ clause.object.entity | lower }}"
{% endif %}
{% if clause.condition %}
    # Condition: {{ clause.condition.trigger }}
    input.condition_met == true
{% endif %}
}

violation[msg] if {
    input.actor == "{{ clause.actor.entity | lower }}"
    not input.{{ clause.action.verb | lower }}_completed
    msg := sprintf("Violation of {{ clause.clause_id }}: %s must {{ clause.action.verb }}", [input.actor])
}

{% elif clause.type == 'prohibition' %}
# Prohibition rule: {{ clause.action.verb }} is forbidden
allow if {
    input.actor == "{{ clause.actor.entity | lower }}"
    not attempted_prohibited_action
}

attempted_prohibited_action if {
    input.action == "{{ clause.action.verb | lower }}"
{% if clause.object %}
    input.object == "{{ clause.object.entity | lower }}"
{% endif %}
}

violation[msg] if {
    input.actor == "{{ clause.actor.entity | lower }}"
    attempted_prohibited_action
    msg := sprintf("Violation of {{ clause.clause_id }}: %s must not {{ clause.action.verb }}", [input.actor])
}

{% else %}
# {{ clause.type | title }} rule
allow if {
    input.actor == "{{ clause.actor.entity | lower }}"
}
{% endif %}

# Metadata
metadata := {
    "clause_id": "{{ clause.clause_id }}",
    "type": "{{ clause.type }}",
    "confidence": {{ confidence }},
    "generated_by": "aegislang-compiler-v{{ version }}",
    "source_text": "{{ clause.source_text | truncate(200) | replace('"', '\\"') }}"
}
'''

JSON_RULE_TEMPLATE = '''{
  "rule_id": "{{ clause.clause_id }}",
  "type": "{{ clause.type }}",
  "version": "{{ version }}",
  "generated_at": "{{ timestamp }}",
  "confidence": {{ confidence }},
  "source": {
    "text": {{ clause.source_text | tojson }},
    "document_id": "{{ doc_id }}"
  },
  "actor": {
    "entity": "{{ clause.actor.entity }}",
    "qualifiers": {{ clause.actor.qualifiers | tojson }},
    "mapped_to": {{ mappings.actor_path | tojson if mappings.actor_path else 'null' }}
  },
  "action": {
    "verb": "{{ clause.action.verb }}",
    "modifiers": {{ clause.action.modifiers | tojson }},
    "enforcement": "{% if clause.type == 'obligation' %}require{% elif clause.type == 'prohibition' %}block{% else %}allow{% endif %}"
  },
{% if clause.object %}
  "object": {
    "entity": "{{ clause.object.entity }}",
    "qualifiers": {{ clause.object.qualifiers | tojson }},
    "mapped_to": {{ mappings.object_path | tojson if mappings.object_path else 'null' }}
  },
{% else %}
  "object": null,
{% endif %}
{% if clause.condition %}
  "condition": {
    "trigger": "{{ clause.condition.trigger }}",
    "temporal": {{ clause.condition.temporal | tojson if clause.condition.temporal else 'null' }}
  },
{% else %}
  "condition": null,
{% endif %}
{% if clause.temporal_scope %}
  "temporal_scope": {
    "deadline": {{ clause.temporal_scope.deadline | tojson if clause.temporal_scope.deadline else 'null' }},
    "frequency": {{ clause.temporal_scope.frequency | tojson if clause.temporal_scope.frequency else 'null' }},
    "duration": {{ clause.temporal_scope.duration | tojson if clause.temporal_scope.duration else 'null' }}
  },
{% else %}
  "temporal_scope": null,
{% endif %}
  "severity": "{{ severity }}",
  "metadata": {
    "generated_by": "aegislang-compiler-v{{ version }}",
    "template": "json_rule"
  }
}'''


# -----------------------------------------------------------------------------
# Template Registry
# -----------------------------------------------------------------------------


class TemplateRegistry:
    """Registry for Jinja2 templates."""

    def __init__(self):
        self._templates: dict[str, dict[str, str]] = {
            "yaml": {
                "obligation": YAML_OBLIGATION_TEMPLATE,
                "prohibition": YAML_PROHIBITION_TEMPLATE,
                "permission": YAML_OBLIGATION_TEMPLATE,
                "conditional": YAML_OBLIGATION_TEMPLATE,
                "definition": YAML_OBLIGATION_TEMPLATE,
                "exception": YAML_OBLIGATION_TEMPLATE,
            },
            "sql": {
                "default": SQL_CHECK_CONSTRAINT_TEMPLATE,
            },
            "python": {
                "default": PYTHON_TEST_TEMPLATE,
            },
            "terraform": {
                "default": TERRAFORM_POLICY_TEMPLATE,
            },
            "rego": {
                "default": REGO_POLICY_TEMPLATE,
            },
            "json": {
                "default": JSON_RULE_TEMPLATE,
            },
        }
        self._env = Environment(loader=BaseLoader())
        self._env.filters["truncate"] = lambda s, length: (
            s[:length] + "..." if len(s) > length else s
        )

    def get_template(
        self, format: ArtifactFormat, clause_type: str
    ) -> tuple[str, str]:
        """
        Get template for format and clause type.

        Returns tuple of (template_string, template_name)
        """
        format_templates = self._templates.get(format.value, {})

        # Try clause-specific template first
        if clause_type in format_templates:
            return format_templates[clause_type], f"{format.value}/{clause_type}"

        # Fall back to default
        if "default" in format_templates:
            return format_templates["default"], f"{format.value}/default"

        raise TemplateNotFound(f"No template for {format.value}/{clause_type}")

    def render(
        self, template_str: str, context: dict[str, Any]
    ) -> str:
        """Render a template with context."""
        template = self._env.from_string(template_str)
        return template.render(**context)

    def register_template(
        self, format: ArtifactFormat, clause_type: str, template: str
    ) -> None:
        """Register a custom template."""
        if format.value not in self._templates:
            self._templates[format.value] = {}
        self._templates[format.value][clause_type] = template

    def load_templates_from_dir(self, templates_dir: Path) -> None:
        """Load templates from directory structure."""
        if not templates_dir.exists():
            return

        for format_dir in templates_dir.iterdir():
            if not format_dir.is_dir():
                continue

            format_name = format_dir.name
            if format_name not in self._templates:
                self._templates[format_name] = {}

            for template_file in format_dir.glob("*.j2"):
                clause_type = template_file.stem
                self._templates[format_name][clause_type] = template_file.read_text()

        logger.info("templates_loaded", directory=str(templates_dir))


# -----------------------------------------------------------------------------
# Syntax Validators
# -----------------------------------------------------------------------------


class SyntaxValidator:
    """Validates syntax of generated artifacts."""

    @staticmethod
    def validate_yaml(content: str) -> tuple[bool, list[str]]:
        """Validate YAML syntax."""
        try:
            import yaml
            yaml.safe_load(content)
            return True, []
        except yaml.YAMLError as e:
            return False, [f"YAML syntax error: {e}"]

    @staticmethod
    def validate_sql(content: str) -> tuple[bool, list[str]]:
        """Validate SQL syntax."""
        try:
            import sqlparse
            parsed = sqlparse.parse(content)
            if not parsed:
                return False, ["Empty SQL"]
            # Basic validation - check for statement types
            warnings = []
            for stmt in parsed:
                if stmt.get_type() == "UNKNOWN":
                    warnings.append(f"Unknown statement type: {str(stmt)[:50]}")
            return True, warnings
        except Exception as e:
            return False, [f"SQL parse error: {e}"]

    @staticmethod
    def validate_python(content: str) -> tuple[bool, list[str]]:
        """Validate Python syntax."""
        try:
            import ast
            ast.parse(content)
            return True, []
        except SyntaxError as e:
            return False, [f"Python syntax error at line {e.lineno}: {e.msg}"]

    @staticmethod
    def validate_json(content: str) -> tuple[bool, list[str]]:
        """Validate JSON syntax."""
        try:
            json.loads(content)
            return True, []
        except json.JSONDecodeError as e:
            return False, [f"JSON syntax error: {e}"]

    @staticmethod
    def validate_rego(content: str) -> tuple[bool, list[str]]:
        """Basic Rego validation (syntax check only)."""
        warnings = []
        # Basic checks
        if "package " not in content:
            warnings.append("Missing package declaration")
        if "import " not in content and "future.keywords" not in content:
            warnings.append("Consider importing future.keywords for modern Rego")
        return True, warnings

    @staticmethod
    def validate_terraform(content: str) -> tuple[bool, list[str]]:
        """Basic Terraform/HCL validation."""
        warnings = []
        # Check for balanced braces
        if content.count("{") != content.count("}"):
            return False, ["Unbalanced braces in HCL"]
        if content.count('"') % 2 != 0:
            return False, ["Unbalanced quotes in HCL"]
        return True, warnings

    def validate(
        self, content: str, format: ArtifactFormat
    ) -> tuple[bool, list[str]]:
        """Validate content for given format."""
        validators = {
            ArtifactFormat.YAML: self.validate_yaml,
            ArtifactFormat.SQL: self.validate_sql,
            ArtifactFormat.PYTHON: self.validate_python,
            ArtifactFormat.JSON: self.validate_json,
            ArtifactFormat.REGO: self.validate_rego,
            ArtifactFormat.TERRAFORM: self.validate_terraform,
        }

        validator = validators.get(format)
        if validator:
            return validator(content)
        return True, []


# -----------------------------------------------------------------------------
# Compiler Agent
# -----------------------------------------------------------------------------


class CompilerAgent:
    """
    L4 Compilation Layer Agent.

    Translates mapped clauses into executable artifacts using templates.
    """

    FILE_EXTENSIONS = {
        ArtifactFormat.YAML: ".yaml",
        ArtifactFormat.SQL: ".sql",
        ArtifactFormat.PYTHON: ".py",
        ArtifactFormat.TERRAFORM: ".tf",
        ArtifactFormat.REGO: ".rego",
        ArtifactFormat.JSON: ".json",
    }

    def __init__(
        self,
        templates_dir: Path | None = None,
        output_dir: Path | None = None,
    ):
        """
        Initialize the compiler agent.

        Args:
            templates_dir: Directory containing custom templates
            output_dir: Directory for output artifacts
        """
        self.template_registry = TemplateRegistry()
        self.validator = SyntaxValidator()
        self.output_dir = output_dir or Path("./artifacts")

        if templates_dir:
            self.template_registry.load_templates_from_dir(templates_dir)

        logger.info("compiler_agent_initialized", output_dir=str(self.output_dir))

    def compile_clause(
        self,
        mapped_clause: dict[str, Any],
        format: ArtifactFormat,
        doc_id: str,
    ) -> CompiledArtifact:
        """
        Compile a single mapped clause to an artifact.

        Args:
            mapped_clause: MappedClause as dictionary
            format: Target output format
            doc_id: Source document ID

        Returns:
            CompiledArtifact
        """
        clause = mapped_clause["source_clause"]
        clause_id = mapped_clause["clause_id"]
        clause_type = clause.get("type", "obligation")

        # Build mapping paths
        mappings = {
            "actor_path": None,
            "object_path": None,
        }
        for entity_mapping in mapped_clause.get("mapped_entities", []):
            role = entity_mapping.get("source_role", "")
            path = entity_mapping.get("target_path", "")
            if role == "actor":
                mappings["actor_path"] = path
            elif role == "object":
                mappings["object_path"] = path

        # Calculate confidence
        confidences = [
            m.get("confidence", 0.5)
            for m in mapped_clause.get("mapped_entities", [])
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Determine severity
        if clause_type == "prohibition":
            severity = "critical"
        elif clause_type == "obligation":
            severity = "high"
        else:
            severity = "medium"

        # Get template
        template_str, template_name = self.template_registry.get_template(
            format, clause_type
        )

        # Build context
        context = {
            "clause": clause,
            "mappings": mappings,
            "doc_id": doc_id,
            "confidence": round(avg_confidence, 2),
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": VERSION,
            # SQL-specific
            "table_name": self._extract_table_name(mappings),
            "check_condition": self._generate_check_condition(clause, mappings),
            # Terraform-specific
            "resource_type": self._infer_resource_type(clause),
        }

        # Render template
        content = self.template_registry.render(template_str, context)

        # Validate syntax
        is_valid, warnings = self.validator.validate(content, format)

        # Generate file path
        ext = self.FILE_EXTENSIONS[format]
        file_path = f"{doc_id}/{clause_id}{ext}"

        # Generate artifact ID
        artifact_id = f"{clause_id}_{format.value}"

        artifact = CompiledArtifact(
            artifact_id=artifact_id,
            clause_id=clause_id,
            format=format,
            content=content,
            file_path=file_path,
            syntax_valid=is_valid,
            template_used=template_name,
            compilation_timestamp=datetime.now(timezone.utc).isoformat(),
            warnings=warnings,
        )

        logger.debug(
            "clause_compiled",
            clause_id=clause_id,
            format=format.value,
            valid=is_valid,
        )

        return artifact

    def _extract_table_name(self, mappings: dict[str, Any]) -> str:
        """Extract table name from mappings."""
        for path in [mappings.get("actor_path"), mappings.get("object_path")]:
            if path and "." in path:
                return path.split(".")[0]
        return "compliance_table"

    def _generate_check_condition(
        self, clause: dict[str, Any], mappings: dict[str, Any]
    ) -> str:
        """Generate SQL check condition from clause."""
        action = clause.get("action", {}).get("verb", "comply")

        # Create a simple check based on action
        action_column = f"{action.lower()}_status"

        if clause.get("type") == "prohibition":
            return f"NOT {action_column}"
        else:
            return f"{action_column} = TRUE"

    def _infer_resource_type(self, clause: dict[str, Any]) -> str:
        """Infer Terraform resource type from clause."""
        actor = clause.get("actor", {}).get("entity", "").lower()

        # Simple mapping of common actors to resource types
        if "storage" in actor or "data" in actor:
            return "aws_s3_bucket"
        elif "network" in actor or "vpc" in actor:
            return "aws_vpc"
        elif "compute" in actor or "server" in actor:
            return "aws_instance"
        elif "database" in actor or "db" in actor:
            return "aws_db_instance"
        else:
            return "aws_resource"

    def compile_mapped_collection(
        self,
        mapped_collection: dict[str, Any],
        formats: list[ArtifactFormat] | None = None,
    ) -> CompiledArtifactCollection:
        """
        Compile all clauses in a mapped collection.

        Args:
            mapped_collection: MappedClauseCollection as dictionary
            formats: List of formats to generate (default: YAML, SQL, Python)

        Returns:
            CompiledArtifactCollection
        """
        if formats is None:
            formats = [ArtifactFormat.YAML, ArtifactFormat.SQL, ArtifactFormat.PYTHON]

        doc_id = mapped_collection["doc_id"]
        artifacts: list[CompiledArtifact] = []

        for clause in mapped_collection.get("clauses", []):
            for format in formats:
                try:
                    artifact = self.compile_clause(clause, format, doc_id)
                    artifacts.append(artifact)
                except Exception as e:
                    logger.warning(
                        "compilation_failed",
                        clause_id=clause.get("clause_id"),
                        format=format.value,
                        error=str(e),
                    )

        collection = CompiledArtifactCollection(
            doc_id=doc_id,
            artifacts=artifacts,
            compilation_timestamp=datetime.now(timezone.utc).isoformat(),
            formats_generated=[f.value for f in formats],
        )

        logger.info(
            "collection_compiled",
            doc_id=doc_id,
            total_artifacts=len(artifacts),
            formats=collection.formats_generated,
        )

        return collection

    def write_artifacts(
        self, collection: CompiledArtifactCollection
    ) -> list[Path]:
        """Write artifacts to output directory."""
        written_paths: list[Path] = []

        for artifact in collection.artifacts:
            output_path = self.output_dir / artifact.file_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(artifact.content)
            written_paths.append(output_path)

        logger.info(
            "artifacts_written",
            count=len(written_paths),
            directory=str(self.output_dir),
        )

        return written_paths


# -----------------------------------------------------------------------------
# Event Publishing (Agent-OS Integration)
# -----------------------------------------------------------------------------


async def publish_compiled_event(
    collection: CompiledArtifactCollection,
    redis_url: str | None = None,
) -> None:
    """Publish policy.compiled event to Agent-OS event bus."""
    if redis_url is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    try:
        import redis.asyncio as redis_async

        client = redis_async.from_url(redis_url)
        await client.publish(
            "policy.compiled",
            collection.model_dump_json(),
        )
        await client.aclose()

        logger.info(
            "event_published",
            topic="policy.compiled",
            doc_id=collection.doc_id,
            artifact_count=len(collection.artifacts),
        )
    except Exception as e:
        logger.warning(
            "event_publish_failed",
            topic="policy.compiled",
            error=str(e),
        )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for compilation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AegisLang Compiler - L4 Compilation Layer"
    )
    parser.add_argument(
        "input",
        help="Input file (JSON from L3 mapper)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./artifacts",
        help="Output directory (default: ./artifacts)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["yaml", "sql", "python", "terraform", "rego", "json"],
        default=["yaml", "sql", "python"],
        help="Output formats to generate (default: yaml sql python)",
    )
    parser.add_argument(
        "--templates",
        help="Custom templates directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Output to stdout instead of files",
    )

    args = parser.parse_args()

    # Initialize agent
    templates_dir = Path(args.templates) if args.templates else None
    output_dir = Path(args.output_dir)

    agent = CompilerAgent(
        templates_dir=templates_dir,
        output_dir=output_dir,
    )

    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        mapped_data = json.loads(input_path.read_text())
        formats = [ArtifactFormat(f) for f in args.formats]

        result = agent.compile_mapped_collection(mapped_data, formats)

        if args.dry_run:
            print(result.model_dump_json(indent=2))
        else:
            paths = agent.write_artifacts(result)
            print(f"Written {len(paths)} artifacts to {output_dir}", file=sys.stderr)
            for path in paths:
                print(f"  - {path}")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
