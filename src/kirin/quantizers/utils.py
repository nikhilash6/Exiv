from typing import get_origin, get_args, Any, Union

def validate_type(value, expected_type, field_name: str):
    """
    Validate that a value matches an expected typing annotation.
    Supports Any, Union, Optional, list, dict, set, and plain types.
    """

    # Case 1: Any → always valid
    if expected_type is Any:
        return

    origin = get_origin(expected_type)

    # Case 2: Non-generic type (int, str, torch.dtype, etc.)
    if origin is None:
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Field '{field_name}' must be {expected_type}, "
                f"but got {type(value)}"
            )
        return

    # Case 3: Union/Optional
    if origin is Union:
        args = get_args(expected_type)
        if not any(
            (
                (t is Any)
                or (get_origin(t) is None and isinstance(value, t))
                or (get_origin(t) is not None and safe_check(value, t))
            )
            for t in args
        ):
            raise TypeError(
                f"Field '{field_name}' must be one of {args}, "
                f"but got {type(value)}"
            )
        return

    # Case 4: list[...] 
    if origin is list:
        (elem_type,) = get_args(expected_type)
        if not isinstance(value, list):
            raise TypeError(f"Field '{field_name}' must be a list, got {type(value)}")
        for v in value:
            validate_type(v, elem_type, f"{field_name} element")
        return

    # Case 5: dict[...]
    if origin is dict:
        key_type, val_type = get_args(expected_type)
        if not isinstance(value, dict):
            raise TypeError(f"Field '{field_name}' must be a dict, got {type(value)}")
        for k, v in value.items():
            validate_type(k, key_type, f"{field_name} key")
            validate_type(v, val_type, f"{field_name} value")
        return

    # Case 6: set[...]
    if origin is set:
        (elem_type,) = get_args(expected_type)
        if not isinstance(value, set):
            raise TypeError(f"Field '{field_name}' must be a set, got {type(value)}")
        for v in value:
            validate_type(v, elem_type, f"{field_name} element")
        return

    # Fallback
    raise TypeError(f"Unsupported type annotation {expected_type} for field '{field_name}'")


def safe_check(value, expected_type):
    """Helper: safe recursive check for nested generics in Union."""
    try:
        validate_type(value, expected_type, "<union member>")
        return True
    except TypeError:
        return False