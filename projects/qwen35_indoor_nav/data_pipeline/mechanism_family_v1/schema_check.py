"""Fail-closed validator for exactly the JSON Schema vocabulary in DATA_SCHEMA_V2.

This is not a general Draft 2020-12 implementation. Unsupported validation
keywords are rejected rather than silently ignored. No external dependency.
"""
import json
import math
import re


ANNOTATIONS = {'$schema', '$id', 'title', 'description', '$defs', 'x-cross-field-assertions'}
KEYWORDS = {'$ref', 'type', 'const', 'enum', 'required', 'properties',
            'additionalProperties', 'items', 'minItems', 'maxItems', 'uniqueItems',
            'minLength', 'pattern', 'minimum', 'maximum', 'allOf', 'anyOf',
            'oneOf', 'if', 'then', 'else'}


def vocabulary_check(schema):
    if isinstance(schema, bool):
        return
    unknown = set(schema) - ANNOTATIONS - KEYWORDS
    if unknown:
        raise ValueError(f'UNSUPPORTED_SCHEMA_KEYWORDS: {sorted(unknown)}')
    for key in ('properties', '$defs'):
        for child in schema.get(key, {}).values():
            vocabulary_check(child)
    for key in ('items', 'additionalProperties', 'if', 'then', 'else'):
        if key in schema:
            vocabulary_check(schema[key])
    for key in ('allOf', 'anyOf', 'oneOf'):
        for child in schema.get(key, []):
            vocabulary_check(child)


def typed(value, name):
    return {'object': lambda: isinstance(value, dict),
            'array': lambda: isinstance(value, list),
            'string': lambda: isinstance(value, str),
            'boolean': lambda: isinstance(value, bool),
            'integer': lambda: isinstance(value, int) and not isinstance(value, bool),
            'number': lambda: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            'null': lambda: value is None}[name]()


def same(a, b):
    # JSON boolean is not numeric 0/1; Python equality alone gets this wrong.
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def validate(value, schema, root=None, path='$'):
    root = schema if root is None else root
    if isinstance(schema, bool):
        if not schema:
            raise ValueError(path + ': false schema')
        return
    if '$ref' in schema:
        ref = schema['$ref']
        if not ref.startswith('#/'):
            raise ValueError('EXTERNAL_SCHEMA_REF_FORBIDDEN')
        child = root
        for key in ref[2:].split('/'):
            child = child[key.replace('~1', '/').replace('~0', '~')]
        validate(value, child, root, path)
    if 'type' in schema:
        types = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        if not any(typed(value, t) for t in types):
            raise ValueError(path + ': type')
    if 'const' in schema and not same(value, schema['const']):
        raise ValueError(path + ': const')
    if 'enum' in schema and not any(same(value, option) for option in schema['enum']):
        raise ValueError(path + ': enum')
    if isinstance(value, dict):
        if set(schema.get('required', [])) - set(value):
            raise ValueError(path + ': required')
        properties = schema.get('properties', {})
        for key, item in value.items():
            validate(item, properties.get(key, schema.get('additionalProperties', True)), root, path + '.' + key)
    if isinstance(value, list):
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', math.inf):
            raise ValueError(path + ': array length')
        if schema.get('uniqueItems') and any(same(x, y) for i, x in enumerate(value) for y in value[i+1:]):
            raise ValueError(path + ': uniqueItems')
        for i, item in enumerate(value):
            validate(item, schema.get('items', True), root, f'{path}[{i}]')
    if isinstance(value, str):
        if len(value) < schema.get('minLength', 0):
            raise ValueError(path + ': minLength')
        if 'pattern' in schema and not re.search(schema['pattern'], value):
            raise ValueError(path + ': pattern')
    if typed(value, 'number'):
        if value < schema.get('minimum', -math.inf) or value > schema.get('maximum', math.inf):
            raise ValueError(path + ': numeric bound')
    for child in schema.get('allOf', []):
        validate(value, child, root, path)
    for key in ('anyOf', 'oneOf'):
        if key in schema:
            n = sum(matches(value, child, root) for child in schema[key])
            if n == 0 or (key == 'oneOf' and n != 1):
                raise ValueError(path + ': ' + key)
    if 'if' in schema:
        branch = 'then' if matches(value, schema['if'], root) else 'else'
        if branch in schema:
            validate(value, schema[branch], root, path)


def matches(value, schema, root):
    try:
        validate(value, schema, root)
        return True
    except ValueError:
        return False
