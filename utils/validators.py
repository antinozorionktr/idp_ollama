"""
JSON Schema Validation Utilities
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from jsonschema import validate, ValidationError, Draft7Validator

logger = logging.getLogger(__name__)


def validate_json_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any]
) -> Tuple[bool, Optional[List[str]]]:
    """
    Validate data against a JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema
        
    Returns:
        Tuple of (is_valid, list of error messages or None)
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        errors = []
        validator = Draft7Validator(schema)
        for error in validator.iter_errors(data):
            path = " -> ".join(str(p) for p in error.absolute_path)
            if path:
                errors.append(f"{path}: {error.message}")
            else:
                errors.append(error.message)
        return False, errors


def coerce_to_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Attempt to coerce data to match a schema.
    
    Handles common type mismatches like string numbers.
    
    Args:
        data: Raw extracted data
        schema: Target schema
        
    Returns:
        Coerced data
    """
    if not isinstance(data, dict):
        return data
    
    properties = schema.get("properties", {})
    result = {}
    
    for key, value in data.items():
        if key in properties:
            prop_schema = properties[key]
            prop_type = prop_schema.get("type")
            
            try:
                if prop_type == "number" and isinstance(value, str):
                    # Try to convert string to number
                    value = value.replace(",", "").replace("$", "").strip()
                    result[key] = float(value) if "." in value else int(value)
                    
                elif prop_type == "integer" and isinstance(value, (str, float)):
                    result[key] = int(float(str(value).replace(",", "")))
                    
                elif prop_type == "string" and not isinstance(value, str):
                    result[key] = str(value)
                    
                elif prop_type == "array" and isinstance(value, dict):
                    # Single item that should be array
                    result[key] = [value]
                    
                elif prop_type == "object" and isinstance(value, dict):
                    # Recursively coerce nested objects
                    result[key] = coerce_to_schema(value, prop_schema)
                    
                elif prop_type == "array" and isinstance(value, list):
                    items_schema = prop_schema.get("items", {})
                    result[key] = [
                        coerce_to_schema(item, items_schema) 
                        if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    result[key] = value
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not coerce {key}: {e}")
                result[key] = value
        else:
            result[key] = value
    
    return result


def extract_required_fields(
    data: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract only the fields defined in the schema.
    
    Args:
        data: Full extracted data
        schema: Schema defining required fields
        
    Returns:
        Filtered data with only schema fields
    """
    properties = schema.get("properties", {})
    result = {}
    
    for key in properties:
        if key in data:
            prop_schema = properties[key]
            value = data[key]
            
            if prop_schema.get("type") == "object" and isinstance(value, dict):
                result[key] = extract_required_fields(value, prop_schema)
            elif prop_schema.get("type") == "array" and isinstance(value, list):
                items_schema = prop_schema.get("items", {})
                if items_schema.get("type") == "object":
                    result[key] = [
                        extract_required_fields(item, items_schema)
                        for item in value if isinstance(item, dict)
                    ]
                else:
                    result[key] = value
            else:
                result[key] = value
    
    return result


def merge_extracted_data(
    existing: Dict[str, Any],
    new: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge extracted data from multiple pages/sources.
    
    Args:
        existing: Existing data
        new: New data to merge
        schema: Optional schema for type-aware merging
        
    Returns:
        Merged data
    """
    result = existing.copy()
    
    for key, value in new.items():
        if key not in result:
            result[key] = value
        elif isinstance(result[key], list) and isinstance(value, list):
            # Extend arrays
            result[key].extend(value)
        elif isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge objects
            result[key] = merge_extracted_data(result[key], value)
        elif result[key] is None:
            result[key] = value
        # Otherwise keep existing value
    
    return result
