"""
Tests for Annotated types with forward references.

This tests the fix for GitHub issue #13056:
https://github.com/fastapi/fastapi/issues/13056

The issue occurs when using `from __future__ import annotations` and defining
a class AFTER an endpoint that uses it in `Annotated[Type, Depends(...)]`.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Annotated, ForwardRef, get_args, get_origin

from fastapi import Depends, FastAPI, Query
from fastapi.dependencies.utils import (
    _parse_annotated_string,
    get_typed_annotation,
)
from fastapi.testclient import TestClient

# --- Module-level dependency functions ---
# These must be at module level to be in globalns when the decorator runs


def get_potato_basic() -> Potato:
    return Potato(color="red", size=10)


def get_potato_schema() -> Potato:
    return Potato(color="blue", size=5)


def get_potato_with_param() -> Potato:
    return Potato(color="green", size=7)


def get_potato_multi() -> Potato:
    return Potato(color="yellow", size=3)


def get_tomato_multi() -> Tomato:
    return Tomato(ripe=True)


def get_potato_nested() -> Potato:
    return Potato(color="purple", size=12)


def get_meal_nested(potato: Annotated[Potato, Depends(get_potato_nested)]) -> Meal:
    return Meal(main=potato)


def get_potato_metadata() -> Potato:
    return Potato(color="orange", size=8)


def get_carrot_normal() -> Carrot:
    return Carrot(length=15)


# --- Forward-referenced types ---


@dataclass
class Potato:
    color: str
    size: int = 0


@dataclass
class Tomato:
    ripe: bool


@dataclass
class Meal:
    main: Potato


@dataclass
class Carrot:
    length: int


# --- Test _parse_annotated_string helper function ---


def test_parse_annotated_with_unresolved_type():
    """Test parsing when the type cannot be resolved."""
    # Simulate globalns that has Depends but not 'UnknownType'
    globalns = {
        "Depends": Depends,
        "get_potato_basic": get_potato_basic,
        "Annotated": Annotated,
    }

    result = _parse_annotated_string(
        "Annotated[UnknownType, Depends(get_potato_basic)]", globalns
    )

    # Should return an Annotated type
    assert result is not None
    assert get_origin(result) is Annotated

    # First arg should be a ForwardRef since UnknownType couldn't be resolved
    args = get_args(result)
    assert isinstance(args[0], ForwardRef)
    assert args[0].__forward_arg__ == "UnknownType"

    # Second arg should be the Depends instance
    assert type(args[1]).__name__ == "Depends"


def test_parse_annotated_with_resolved_type():
    """Test parsing when the type can be resolved."""
    globalns = {
        "Depends": Depends,
        "get_potato_basic": get_potato_basic,
        "Potato": Potato,
        "Annotated": Annotated,
    }

    result = _parse_annotated_string(
        "Annotated[Potato, Depends(get_potato_basic)]", globalns
    )

    assert result is not None
    assert get_origin(result) is Annotated

    args = get_args(result)
    assert args[0] is Potato
    assert type(args[1]).__name__ == "Depends"


def test_parse_annotated_with_multiple_metadata():
    """Test parsing with multiple metadata items."""
    globalns = {
        "Annotated": Annotated,
        "Depends": Depends,
        "get_potato_basic": get_potato_basic,
    }

    result = _parse_annotated_string(
        'Annotated[UnknownType, "doc string", Depends(get_potato_basic)]', globalns
    )

    assert result is not None
    assert get_origin(result) is Annotated

    args = get_args(result)
    assert isinstance(args[0], ForwardRef)
    assert args[1] == "doc string"
    assert type(args[2]).__name__ == "Depends"


def test_parse_non_annotated_returns_none():
    """Test that non-Annotated strings return None."""
    globalns = {}

    result = _parse_annotated_string("SomeOtherType", globalns)
    assert result is None

    result = _parse_annotated_string("List[int]", globalns)
    assert result is None


def test_parse_annotated_no_comma_returns_none():
    """Annotated without metadata returns None."""
    globalns = {}

    # This is technically invalid Annotated usage, but test the edge case
    result = _parse_annotated_string("Annotated[SomeType]", globalns)
    assert result is None


# --- Test get_typed_annotation with ForwardRef ---


def test_string_annotation_with_unresolved_annotated():
    """Test that string annotation with unresolved Annotated type is handled."""
    # Simulate a namespace where the type is not defined but Depends is available
    globalns = {
        "Annotated": Annotated,
        "Depends": Depends,
        "get_potato_basic": get_potato_basic,
    }

    # This simulates what happens with `from __future__ import annotations`
    # when Potato is not yet defined
    result = get_typed_annotation(
        "Annotated[Potato, Depends(get_potato_basic)]", globalns
    )

    # Should return an Annotated type, not a ForwardRef
    assert get_origin(result) is Annotated

    # The first arg should be a ForwardRef('Potato')
    args = get_args(result)
    assert isinstance(args[0], ForwardRef)
    assert args[0].__forward_arg__ == "Potato"

    # The Depends should be extractable
    assert type(args[1]).__name__ == "Depends"


def test_string_annotation_with_resolved_type():
    """Test normal case where type resolves successfully."""
    globalns = {
        "Annotated": Annotated,
        "Depends": Depends,
        "get_potato_basic": get_potato_basic,
        "Potato": Potato,
    }

    result = get_typed_annotation(
        "Annotated[Potato, Depends(get_potato_basic)]", globalns
    )

    # Should return the properly resolved Annotated type
    assert get_origin(result) is Annotated
    args = get_args(result)
    assert args[0] is Potato
    assert type(args[1]).__name__ == "Depends"


def test_aliased_annotated_import():
    """Test that aliased Annotated imports are detected and handled.

    When users do `from typing import Annotated as Ann`, the annotation
    string will be 'Ann[Type, Depends(...)]' instead of 'Annotated[...]'.
    """
    from fastapi.dependencies.utils import _is_annotated_string

    # Simulate namespace with aliased Annotated
    globalns = {
        "Ann": Annotated,  # Aliased import
        "Annotated": typing.Annotated,
        "Depends": Depends,
        "get_potato_basic": get_potato_basic,
        "typing": typing,
    }

    # Test _is_annotated_string detects the alias
    assert _is_annotated_string("Ann[Potato, Depends(get_potato_basic)]", globalns)
    assert _is_annotated_string(
        "Annotated[Potato, Depends(get_potato_basic)]", globalns
    )
    assert _is_annotated_string("typing.Annotated[Potato, Depends()]", globalns)
    assert not _is_annotated_string("List[int]", globalns)
    assert not _is_annotated_string("SomeOther[Type]", globalns)

    # Test that get_typed_annotation handles aliased Annotated
    result = get_typed_annotation("Ann[Potato, Depends(get_potato_basic)]", globalns)

    # Should return an Annotated type
    assert get_origin(result) is Annotated

    # First arg should be ForwardRef since Potato isn't in globalns
    args = get_args(result)
    assert isinstance(args[0], ForwardRef)
    assert args[0].__forward_arg__ == "Potato"

    # Depends should be extracted
    assert type(args[1]).__name__ == "Depends"


def test_is_annotated_string_handles_unsafe_expressions():
    """Test that _is_annotated_string gracefully handles expressions that trigger ValueError.

    When the expression before '[' contains unsafe AST nodes (like lambda),
    _safe_eval raises ValueError. The function should catch this and return False
    instead of crashing.
    """
    from fastapi.dependencies.utils import _is_annotated_string

    globalns = {"Annotated": Annotated}

    # Lambda expressions trigger ValueError in _safe_eval due to unsafe AST node
    # This tests that ValueError is properly caught
    result = _is_annotated_string("(lambda: Annotated)[Type, meta]", globalns)
    assert result is False

    # Expression with unsafe function name triggers ValueError
    result = _is_annotated_string("exec[Type, meta]", globalns)
    assert result is False

    # Expression with unsafe attribute access triggers ValueError
    result = _is_annotated_string("obj.__globals__[Type, meta]", globalns)
    assert result is False


def test_get_typed_annotation_handles_none_type():
    """Test that None type annotations are handled correctly.

    This tests the code path: `if annotation is type(None): return None`
    in get_typed_annotation, verifying it's reachable and works correctly.

    Note: The `type(None)` check is only reachable when the annotation starts
    as a ForwardRef that evaluates to `type(None)`. This happens when:
    1. A string "None" is passed
    2. A ForwardRef("None") is passed
    And the globalns contains "None": type(None).
    """
    # Test with string "None" that resolves to type(None)
    # This is the most common case with `from __future__ import annotations`
    result = get_typed_annotation("None", {"None": type(None)})
    assert result is None

    # Test with ForwardRef to "None" that resolves to type(None)
    result = get_typed_annotation(ForwardRef("None"), {"None": type(None)})
    assert result is None

    # Test edge case: type(None) passed directly is returned as-is
    # (not converted to None, as it doesn't go through ForwardRef evaluation)
    result = get_typed_annotation(type(None), {})
    assert result is type(None)


# --- Integration tests: Annotated[ForwardRef, Depends(...)] ---


def test_basic_depends_with_forward_ref():
    """Basic case: endpoint uses forward-referenced type in Depends."""
    app = FastAPI()

    @app.get("/")
    async def read_root(
        potato: Annotated[Potato, Depends(get_potato_basic)],
    ):
        return {"color": potato.color, "size": potato.size}

    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"color": "red", "size": 10}


def test_openapi_schema_generates():
    """OpenAPI schema should generate without errors."""
    app = FastAPI()

    @app.get("/")
    async def read_root(
        potato: Annotated[Potato, Depends(get_potato_schema)],
    ):
        return {"color": potato.color, "size": potato.size}

    client = TestClient(app)

    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    # Verify the endpoint exists
    assert "/" in schema["paths"]
    get_operation = schema["paths"]["/"]["get"]

    # Verify NO query parameters (potato is a dependency, not a query param)
    assert (
        "parameters" not in get_operation
        or len(get_operation.get("parameters", [])) == 0
    )


def test_depends_with_forward_ref_and_regular_params():
    """Forward ref dependency combined with regular query params."""
    app = FastAPI()

    @app.get("/")
    async def read_root(
        potato: Annotated[Potato, Depends(get_potato_with_param)],
        name: str = "default",
    ):
        return {"color": potato.color, "name": name}

    client = TestClient(app)

    # Test with default
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"color": "green", "name": "default"}

    # Test with query param
    response = client.get("/?name=custom")
    assert response.status_code == 200
    assert response.json() == {"color": "green", "name": "custom"}

    # OpenAPI should show only the 'name' query param
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    params = schema["paths"]["/"]["get"].get("parameters", [])
    assert len(params) == 1
    assert params[0]["name"] == "name"
    assert params[0]["in"] == "query"


# --- Test Annotated[ForwardRef, Query(...)] scenarios ---


def test_query_with_annotated_type():
    """Query parameter with Annotated should still work."""
    app = FastAPI()

    @app.get("/")
    async def read_root(value: Annotated[int, Query(description="A value")]):
        return {"value": value}

    client = TestClient(app)

    response = client.get("/?value=42")
    assert response.status_code == 200
    assert response.json() == {"value": 42}


# --- Test multiple dependencies with forward refs ---


def test_multiple_depends_with_forward_refs():
    """Multiple dependencies each with forward ref types."""
    app = FastAPI()

    @app.get("/")
    async def read_root(
        potato: Annotated[Potato, Depends(get_potato_multi)],
        tomato: Annotated[Tomato, Depends(get_tomato_multi)],
    ):
        return {"potato_color": potato.color, "tomato_ripe": tomato.ripe}

    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"potato_color": "yellow", "tomato_ripe": True}

    # OpenAPI should have no query params
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    get_op = schema["paths"]["/"]["get"]
    assert "parameters" not in get_op or len(get_op.get("parameters", [])) == 0


# --- Test nested dependencies with forward refs ---


def test_nested_dependency_with_forward_ref():
    """Dependency that itself has a forward ref dependency."""
    app = FastAPI()

    @app.get("/")
    async def read_root(meal: Annotated[Meal, Depends(get_meal_nested)]):
        return {"main_color": meal.main.color}

    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"main_color": "purple"}


# --- Edge cases ---


def test_annotated_with_multiple_metadata():
    """Annotated with multiple metadata items."""
    app = FastAPI()

    # Multiple metadata in Annotated (only Depends should be used)
    @app.get("/")
    async def read_root(
        potato: Annotated[Potato, "some doc", Depends(get_potato_metadata)],
    ):
        return {"color": potato.color}

    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"color": "orange"}


def test_class_defined_before_works_as_expected():
    """Verify normal case (class before endpoint) still works."""
    app = FastAPI()

    @app.get("/")
    async def read_root(carrot: Annotated[Carrot, Depends(get_carrot_normal)]):
        return {"length": carrot.length}

    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"length": 15}

    # OpenAPI works
    response = client.get("/openapi.json")
    assert response.status_code == 200


# --- Security tests for _safe_eval ---


def test_safe_eval_blocks_builtins_name():
    """Test that _safe_eval blocks __builtins__ as a name.

    This is a critical security test. __builtins__ is accessible as an ast.Name
    node (a variable in globalns), not just as an attribute. An attacker could
    use __builtins__.exec("...") to execute arbitrary code if not blocked.
    """
    import pytest
    from fastapi.dependencies.utils import _safe_eval

    globalns = {"__builtins__": __builtins__}

    # Direct access to __builtins__ should be blocked
    with pytest.raises(ValueError, match="Unsafe function name __builtins__"):
        _safe_eval("__builtins__", globalns)


def test_safe_eval_blocks_builtins_exec():
    """Test that _safe_eval blocks __builtins__.exec access.

    This verifies that attempting to access exec via __builtins__ is blocked.
    The __builtins__ name itself should be caught first.
    """
    import pytest
    from fastapi.dependencies.utils import _safe_eval

    globalns = {"__builtins__": __builtins__}

    # Attempting to access __builtins__.exec should be blocked
    # (The __builtins__ name is caught first, before the .exec attribute)
    with pytest.raises(ValueError, match="Unsafe function name __builtins__"):
        _safe_eval("__builtins__.exec", globalns)


# --- Comprehensive security tests for _safe_eval ---


def test_safe_eval_blocks_dangerous_builtins():
    """Test that _safe_eval blocks dangerous function names.

    This is a critical security test ensuring that common code execution
    primitives cannot be invoked through _safe_eval.
    """
    import pytest
    from fastapi.dependencies.utils import _safe_eval

    globalns = {
        "__import__": __import__,
        "exec": exec,
        "eval": eval,
        "compile": compile,
        "open": open,
        "__builtins__": __builtins__,
    }

    # Test __import__('os') - blocks module imports
    with pytest.raises(ValueError, match="Unsafe function name __import__"):
        _safe_eval("__import__('os')", globalns)

    # Test exec('print(1)') - blocks code execution
    with pytest.raises(ValueError, match="Unsafe function name exec"):
        _safe_eval("exec('print(1)')", globalns)

    # Test eval('1+1') - blocks dynamic evaluation
    with pytest.raises(ValueError, match="Unsafe function name eval"):
        _safe_eval("eval('1+1')", globalns)

    # Test compile('x', '', 'exec') - blocks code compilation
    with pytest.raises(ValueError, match="Unsafe function name compile"):
        _safe_eval("compile('x', '', 'exec')", globalns)

    # Test open('/etc/passwd') - blocks file access
    with pytest.raises(ValueError, match="Unsafe function name open"):
        _safe_eval("open('/etc/passwd')", globalns)

    # Test __builtins__ access (for completeness, also tested above)
    with pytest.raises(ValueError, match="Unsafe function name __builtins__"):
        _safe_eval("__builtins__", globalns)


def test_safe_eval_blocks_dunder_attribute_access():
    """Test that _safe_eval blocks dangerous dunder attribute access.

    These attributes can be used to escape the sandbox and access
    arbitrary objects/code. This is critical for security.
    """
    import pytest
    from fastapi.dependencies.utils import _safe_eval

    globalns = {"obj": object(), "x": lambda: None}

    # Test ().__class__.__bases__[0] - accessing base classes
    # Note: ast.walk may encounter __bases__ or __class__ first, so match either
    with pytest.raises(ValueError, match="Unsafe attribute (__class__|__bases__)"):
        _safe_eval("().__class__.__bases__[0]", globalns)

    # Test ().__class__.__mro__ - accessing method resolution order
    # Note: ast.walk may encounter __mro__ or __class__ first, so match either
    with pytest.raises(ValueError, match="Unsafe attribute (__class__|__mro__)"):
        _safe_eval("().__class__.__mro__", globalns)

    # Test str.__subclasses__() - listing subclasses (can find exploitable classes)
    with pytest.raises(ValueError, match="Unsafe attribute __subclasses__"):
        _safe_eval("str.__subclasses__()", globalns)

    # Test x.__globals__ - accessing function's global namespace
    with pytest.raises(ValueError, match="Unsafe attribute __globals__"):
        _safe_eval("x.__globals__", globalns)

    # Test obj.__builtins__ - accessing builtins via attribute
    with pytest.raises(ValueError, match="Unsafe attribute __builtins__"):
        _safe_eval("obj.__builtins__", globalns)

    # Test obj.__code__ - accessing code objects
    with pytest.raises(ValueError, match="Unsafe attribute __code__"):
        _safe_eval("obj.__code__", globalns)


def test_safe_eval_blocks_unsafe_ast_nodes():
    """Test that _safe_eval blocks unsafe AST node types.

    Lambda expressions and comprehensions can be used to execute
    arbitrary code and should be blocked.
    """
    import pytest
    from fastapi.dependencies.utils import _safe_eval

    globalns = {"range": range}

    # Test lambda expression - can execute arbitrary code
    with pytest.raises(ValueError, match="Unsafe node type Lambda"):
        _safe_eval("lambda: 1", globalns)

    # Test list comprehension - can execute code via iteration
    with pytest.raises(ValueError, match="Unsafe node type ListComp"):
        _safe_eval("[x for x in range(10)]", globalns)

    # Test dict comprehension - can execute code via iteration
    with pytest.raises(ValueError, match="Unsafe node type DictComp"):
        _safe_eval("{x: x for x in range(5)}", globalns)


def test_safe_eval_raises_on_invalid_syntax():
    """Test that _safe_eval raises ValueError on syntax errors.

    Invalid syntax should be caught and converted to ValueError,
    not propagate as SyntaxError.
    """
    import pytest
    from fastapi.dependencies.utils import _safe_eval

    globalns = {}

    # Test unclosed bracket
    with pytest.raises(ValueError, match="Invalid syntax"):
        _safe_eval("Annotated[Type,", globalns)

    # Test unclosed parenthesis
    with pytest.raises(ValueError, match="Invalid syntax"):
        _safe_eval("Depends((", globalns)


def test_safe_eval_blocks_closure_access():
    """Test that _safe_eval blocks __closure__ access.

    __closure__ can expose captured variables from closures which may
    contain sensitive data like credentials or API keys.
    """
    import pytest
    from fastapi.dependencies.utils import _safe_eval

    globalns = {"x": lambda: None}

    with pytest.raises(ValueError, match="Unsafe attribute __closure__"):
        _safe_eval("x.__closure__", globalns)


def test_safe_eval_blocks_generator_expression():
    """Test that _safe_eval blocks generator expressions.

    Generator expressions can execute arbitrary code during iteration.
    """
    import pytest
    from fastapi.dependencies.utils import _safe_eval

    globalns = {"range": range}

    with pytest.raises(ValueError, match="Unsafe node type GeneratorExp"):
        _safe_eval("(x for x in range(10))", globalns)


def test_safe_eval_blocks_set_comprehension():
    """Test that _safe_eval blocks set comprehensions.

    Like list/dict comprehensions, set comprehensions can execute
    arbitrary code during iteration.
    """
    import pytest
    from fastapi.dependencies.utils import _safe_eval

    globalns = {"range": range}

    with pytest.raises(ValueError, match="Unsafe node type SetComp"):
        _safe_eval("{x for x in range(5)}", globalns)
