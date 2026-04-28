"""Tests for schema-driven extraction (issue #45)."""

from mantis_agent.extraction import ClaudeExtractor, ExtractionResult, ExtractionSchema
from mantis_agent.graph.objective import ObjectiveSpec, OutputField


# ── ExtractionSchema ──


def test_default_boattrader_schema():
    schema = ExtractionSchema.default_boattrader()
    assert schema.entity_name == "boat listing"
    assert schema.required_fields == ["year", "make"]
    assert "marinemax" in schema.spam_indicators
    assert "marine" in schema.spam_seller_indicators
    assert schema.spam_label == "dealer"
    assert len(schema.fields) == 7


def test_from_objective_zillow():
    objective = ObjectiveSpec(
        raw_text="Find houses for sale",
        domains=["zillow.com"],
        target_entity="property listing",
        forbidden_actions=["Contact Agent", "Request Tour"],
        allowed_reveal_actions=["Show more", "View details"],
        output_schema=[
            OutputField(name="address", required=True, example="123 Main St"),
            OutputField(name="price", required=True, example="$450,000"),
            OutputField(name="beds", required=False, example="3"),
            OutputField(name="baths", required=False, example="2"),
        ],
    )
    schema = ExtractionSchema.from_objective(objective)
    assert schema.entity_name == "property listing"
    assert schema.required_fields == ["address", "price"]
    assert "Contact Agent" in schema.forbidden_controls
    assert "View details" in schema.allowed_controls
    assert len(schema.fields) == 4


def test_from_objective_indeed():
    objective = ObjectiveSpec(
        raw_text="Find software engineer jobs",
        domains=["indeed.com"],
        target_entity="job posting",
        output_schema=[
            OutputField(name="title", required=True, example="Senior Engineer"),
            OutputField(name="company", required=True, example="Acme Corp"),
            OutputField(name="salary", required=False, example="$150,000"),
            OutputField(name="location", required=False, example="San Francisco, CA"),
        ],
    )
    schema = ExtractionSchema.from_objective(objective)
    assert schema.entity_name == "job posting"
    assert schema.required_fields == ["title", "company"]


def test_from_objective_no_output_schema():
    objective = ObjectiveSpec(raw_text="Search for items", domains=["example.com"])
    schema = ExtractionSchema.from_objective(objective)
    assert schema.entity_name == "listing"
    assert "url" in [f["name"] for f in schema.fields]


def test_schema_field_names():
    schema = ExtractionSchema.default_boattrader()
    assert schema.field_names() == ["year", "make", "model", "price", "phone", "url", "seller"]


def test_schema_json_template():
    schema = ExtractionSchema(
        fields=[
            {"name": "title", "type": "str"},
            {"name": "price", "type": "str"},
        ]
    )
    template = schema.json_template()
    assert '"title": ""' in template
    assert '"is_spam": false' in template


def test_schema_field_descriptions():
    schema = ExtractionSchema(
        entity_name="job",
        spam_label="recruiter",
        fields=[
            {"name": "title", "type": "str", "required": True, "example": "Engineer"},
            {"name": "salary", "type": "str", "required": False},
        ],
    )
    desc = schema.field_descriptions()
    assert "title" in desc
    assert "[REQUIRED]" in desc
    assert "Engineer" in desc
    assert "recruiter" in desc


def test_schema_spam_detection():
    schema = ExtractionSchema(
        spam_indicators=["sponsored", "premium"],
        spam_seller_indicators=["agency", "inc"],
    )
    assert schema.contains_spam_text("This is a Sponsored listing")
    assert not schema.contains_spam_text("Normal listing")
    assert schema.seller_looks_like_spam("Talent Agency LLC")
    assert not schema.seller_looks_like_spam("John Smith")


# ── ExtractionResult with schema ──


def test_result_with_schema_viable():
    schema = ExtractionSchema(
        required_fields=["address", "price"],
    )
    result = ExtractionResult(
        _schema=schema,
        extracted_fields={"address": "123 Main", "price": "$500k", "beds": "3"},
    )
    assert result.is_viable()
    assert result.missing_required_reason() == ""


def test_result_with_schema_not_viable_missing_required():
    schema = ExtractionSchema(
        required_fields=["address", "price"],
    )
    result = ExtractionResult(
        _schema=schema,
        extracted_fields={"beds": "3"},
    )
    assert not result.is_viable()
    assert "address" in result.missing_required_reason()
    assert "price" in result.missing_required_reason()


def test_result_with_schema_spam_detection():
    schema = ExtractionSchema(
        required_fields=["title"],
        spam_indicators=["sponsored"],
    )
    result = ExtractionResult(
        _schema=schema,
        extracted_fields={"title": "Test"},
        raw_response="This is a Sponsored listing",
    )
    assert not result.is_private_seller()
    assert "spam" not in result.dealer_reason() or "indicator" in result.dealer_reason()


def test_result_with_schema_to_summary():
    schema = ExtractionSchema(
        fields=[
            {"name": "address", "type": "str"},
            {"name": "price", "type": "str"},
            {"name": "phone", "type": "str"},
        ],
    )
    result = ExtractionResult(
        _schema=schema,
        extracted_fields={"address": "123 Main", "price": "$500k", "phone": ""},
    )
    summary = result.to_summary()
    assert summary.startswith("VIABLE")
    assert "Address: 123 Main" in summary
    assert "Price: $500k" in summary
    assert "Phone: none" in summary


def test_result_without_schema_backward_compat():
    result = ExtractionResult(year="2020", make="Sea Ray", model="240")
    assert result.is_viable()
    assert "Year: 2020" in result.to_summary()
    assert result.missing_required_reason() == ""


def test_result_without_schema_not_viable():
    result = ExtractionResult(make="Sea Ray")
    assert not result.is_viable()
    assert "year" in result.missing_required_reason()


# ── ClaudeExtractor with schema ──


def test_extractor_no_schema_uses_legacy_prompt():
    extractor = ClaudeExtractor()
    prompt = extractor._get_extract_prompt()
    assert "boat listing" in prompt.lower()


def test_extractor_with_schema_uses_dynamic_prompt():
    schema = ExtractionSchema(
        entity_name="property listing",
        fields=[
            {"name": "address", "type": "str", "required": True, "example": "123 Main"},
            {"name": "price", "type": "str", "required": True, "example": "$450k"},
        ],
    )
    extractor = ClaudeExtractor(schema=schema)
    prompt = extractor._get_extract_prompt()
    assert "property listing" in prompt
    assert "address" in prompt
    assert "boat" not in prompt.lower()


def test_extractor_dynamic_multi_prompt():
    schema = ExtractionSchema(
        entity_name="job posting",
        spam_label="recruiter",
        spam_indicators=["staffing agency"],
        fields=[{"name": "title", "type": "str"}],
    )
    extractor = ClaudeExtractor(schema=schema)
    prompt = extractor._get_multi_extract_prompt()
    assert "job posting" in prompt
    assert "recruiter" in prompt.lower()


def test_extractor_dynamic_content_control_prompt():
    schema = ExtractionSchema(
        entity_name="listing",
        allowed_controls=["Show more", "Expand"],
        forbidden_controls=["Contact Agent"],
    )
    extractor = ClaudeExtractor(schema=schema)
    prompt = extractor._get_content_control_prompt()
    assert "Show more" in prompt
    assert "Contact Agent" in prompt


def test_extractor_parse_schema_result():
    schema = ExtractionSchema(
        fields=[
            {"name": "address", "type": "str"},
            {"name": "price", "type": "str"},
            {"name": "phone", "type": "str"},
        ],
        required_fields=["address"],
    )
    extractor = ClaudeExtractor(schema=schema)
    result = extractor._parse_schema_result({
        "address": "123 Main St",
        "price": "$500,000",
        "phone": "305-555-1234",
        "is_spam": False,
    })
    assert result.extracted_fields["address"] == "123 Main St"
    assert result.extracted_fields["price"] == "$500,000"
    assert result.is_viable()
    assert result._schema is schema
