"""
Validators for JSON Schema enforcement
"""

import jsonschema
from jsonschema import validate, ValidationError
from typing import Dict, Any
import logging
import json

logger = logging.getLogger(__name__)


def validate_json_output(
    data: Dict[str, Any], 
    schema: Any
) -> Dict[str, Any]:
    """
    Validate extracted data against JSON schema
    
    Args:
        data: Extracted data dictionary
        schema: JSON schema (string or dict)
        
    Returns:
        Validated data (same as input if valid)
        
    Raises:
        ValidationError: If data doesn't match schema
    """
    try:
        # Parse schema if string
        if isinstance(schema, str):
            schema = json.loads(schema)
        
        # Validate
        validate(instance=data, schema=schema)
        
        logger.info("JSON validation successful")
        return data
        
    except ValidationError as e:
        logger.error(f"JSON validation failed: {str(e)}")
        raise ValueError(f"Data does not match schema: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid schema: {str(e)}")
        raise ValueError(f"Invalid JSON schema: {str(e)}")


def create_example_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Example schemas for common document types
    
    Returns:
        Dictionary of schema name to schema definition
    """
    return {
        "invoice": {
            "type": "object",
            "properties": {
                "invoice_number": {"type": "string"},
                "date": {"type": "string", "format": "date"},
                "vendor": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"},
                        "email": {"type": "string", "format": "email"}
                    },
                    "required": ["name"]
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                            "total": {"type": "number"}
                        },
                        "required": ["description", "quantity", "unit_price"]
                    }
                },
                "subtotal": {"type": "number"},
                "tax": {"type": "number"},
                "total": {"type": "number"}
            },
            "required": ["invoice_number", "date", "vendor", "items", "total"]
        },
        
        "contract": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "title": {"type": "string"},
                "effective_date": {"type": "string", "format": "date"},
                "expiration_date": {"type": "string", "format": "date"},
                "parties": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                            "address": {"type": "string"}
                        },
                        "required": ["name", "role"]
                    }
                },
                "terms": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "payment_terms": {"type": "string"},
                "renewal_terms": {"type": "string"}
            },
            "required": ["contract_id", "effective_date", "parties"]
        },
        
        "resume": {
            "type": "object",
            "properties": {
                "personal_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                        "phone": {"type": "string"},
                        "location": {"type": "string"}
                    },
                    "required": ["name", "email"]
                },
                "summary": {"type": "string"},
                "experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "company": {"type": "string"},
                            "location": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["title", "company"]
                    }
                },
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "degree": {"type": "string"},
                            "institution": {"type": "string"},
                            "graduation_date": {"type": "string"}
                        },
                        "required": ["degree", "institution"]
                    }
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["personal_info", "experience"]
        },
        
        "medical_report": {
            "type": "object",
            "properties": {
                "patient_info": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "date_of_birth": {"type": "string", "format": "date"},
                        "patient_id": {"type": "string"}
                    },
                    "required": ["name"]
                },
                "report_date": {"type": "string", "format": "date"},
                "diagnosis": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "medications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "dosage": {"type": "string"},
                            "frequency": {"type": "string"}
                        },
                        "required": ["name"]
                    }
                },
                "lab_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "test_name": {"type": "string"},
                            "result": {"type": "string"},
                            "reference_range": {"type": "string"}
                        }
                    }
                },
                "notes": {"type": "string"}
            },
            "required": ["patient_info", "report_date"]
        },
        
        "financial_statement": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "period": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"}
                    },
                    "required": ["start_date", "end_date"]
                },
                "revenue": {"type": "number"},
                "expenses": {
                    "type": "object",
                    "properties": {
                        "operating": {"type": "number"},
                        "administrative": {"type": "number"},
                        "other": {"type": "number"}
                    }
                },
                "net_income": {"type": "number"},
                "assets": {
                    "type": "object",
                    "properties": {
                        "current": {"type": "number"},
                        "fixed": {"type": "number"},
                        "total": {"type": "number"}
                    }
                },
                "liabilities": {
                    "type": "object",
                    "properties": {
                        "current": {"type": "number"},
                        "long_term": {"type": "number"},
                        "total": {"type": "number"}
                    }
                }
            },
            "required": ["company_name", "period", "revenue", "net_income"]
        }
    }


def get_schema_by_name(schema_name: str) -> Dict[str, Any]:
    """
    Get a predefined schema by name
    
    Args:
        schema_name: Name of the schema (e.g., 'invoice', 'contract')
        
    Returns:
        Schema definition
    """
    schemas = create_example_schemas()
    
    if schema_name not in schemas:
        available = ", ".join(schemas.keys())
        raise ValueError(
            f"Schema '{schema_name}' not found. "
            f"Available schemas: {available}"
        )
    
    return schemas[schema_name]
