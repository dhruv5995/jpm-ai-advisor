import pytest
from pydantic import ValidationError

from jpm_ai_advisor.protocol import ClientReply, DelegateToAnalyst, to_tool_schema


def test_to_tool_schema_strips_title_and_carries_description() -> None:
    schema = to_tool_schema(ClientReply, "reply", "Send your reply to the advisor.")
    assert schema.name == "reply"
    assert schema.description == "Send your reply to the advisor."
    assert "title" not in schema.input_schema
    assert set(schema.input_schema["required"]) == {"message", "satisfied"}


def test_model_validate_rejects_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        DelegateToAnalyst.model_validate({"context": "no task given"})


def test_model_validate_accepts_well_formed_payload() -> None:
    reply = ClientReply.model_validate({"message": "Sounds good.", "satisfied": True})
    assert reply.satisfied is True
