"""CPU information-boundary contracts, not a learned memory implementation."""
from dataclasses import dataclass

MOVES = ('move_forward', 'turn_left', 'turn_right')


@dataclass(frozen=True)
class PolicyObservation:
    instruction: str
    rgb: tuple
    executed: tuple

    def __post_init__(self):
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError('INSTRUCTION_REQUIRED')
        if not isinstance(self.rgb, tuple) or not 1 <= len(self.rgb) <= 2:
            raise ValueError('RGB_WINDOW')
        if any(type(x) is not bytes or len(x) != 224*224*3 for x in self.rgb):
            raise ValueError('RAW_RGB_REQUIRED_NOT_ID_OR_HASH')
        if not isinstance(self.executed, tuple) or len(self.executed) > 8:
            raise ValueError('EXECUTED_WINDOW')
        if any(x not in MOVES for x in self.executed):
            raise ValueError('ONLY_ACTUALLY_EXECUTED_MOTIONS')


@dataclass(frozen=True)
class PrefixTrace:
    observations: tuple

    def __post_init__(self):
        if not self.observations or not all(type(o) is PolicyObservation for o in self.observations):
            raise ValueError('FULL_CAUSAL_PREFIX_REQUIRED')
        first = self.observations[0]
        if first.executed or len(first.rgb) != 1:
            raise ValueError('PREFIX_MUST_START_AT_RESET')
        for before, now in zip(self.observations, self.observations[1:]):
            if now.instruction != first.instruction:
                raise ValueError('TASK_CHANGE_REQUIRES_PREFIX_RECOMPUTATION')
            if len(now.rgb) != 2 or now.rgb[0] != before.rgb[-1]:
                raise ValueError('NONCAUSAL_RGB_SEQUENCE')
            if len(now.executed) != min(8, len(before.executed)+1) or now.executed[:-1] != before.executed[-7:]:
                raise ValueError('EXECUTED_HISTORY_NOT_WRITTEN_BACK')


@dataclass(frozen=True)
class LabelCertificate:
    outcome: str
    y: object
    mask: int

    def __post_init__(self):
        expected = {'PASS': (1, 1), 'FAIL': (0, 1), 'UNKNOWN': (None, 0)}
        if self.outcome not in expected or (self.y, self.mask) != expected[self.outcome]:
            raise ValueError('UNKNOWN_IS_NOT_A_NEGATIVE')
        if type(self.mask) is not int or self.y is not None and type(self.y) is not int:
            raise ValueError('LABEL_TYPE')


@dataclass(frozen=True)
class FamilySplit:
    split: str
    house_id: str
    physical_family_id: str
    physical_routes: tuple
    history_lineages: tuple
    continuation_lineages: tuple
    language_families: tuple
    exposure: str

    def __post_init__(self):
        if self.split not in ('train', 'dev', 'test') or self.exposure not in ('already_exposed', 'unexposed_provenance_pending', 'independent_provenance_verified'):
            raise ValueError('SPLIT_OR_EXPOSURE')
        for field in ('house_id', 'physical_family_id'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError('GROUP_ID_REQUIRED')
        for field in ('physical_routes', 'history_lineages', 'continuation_lineages', 'language_families'):
            values = getattr(self, field)
            if not isinstance(values, tuple) or not values or any(not isinstance(x, str) or not x for x in values):
                raise ValueError('COMPLETE_GROUP_LINEAGES_REQUIRED')
        if self.split == 'test' and self.exposure != 'independent_provenance_verified':
            raise ValueError('CANNOT_RENAME_EXPOSED_DATA_AS_INDEPENDENT_TEST')


def validate_splits(families):
    seen = {}
    for family in families:
        keys = [('house', family.house_id), ('family', family.physical_family_id)]
        # Histories/continuations share a trajectory namespace to catch role swaps.
        keys += [('trajectory', x) for x in family.physical_routes + family.history_lineages + family.continuation_lineages]
        keys += [('language_family', x) for x in family.language_families]
        for key in keys:
            if seen.setdefault(key, family.split) != family.split:
                raise ValueError('CROSS_SPLIT_FAMILY_LEAKAGE:' + key[0])


def from_existing_v4(loader, record):
    """Resolve and verify real RGB with existing loader, then drop transport IDs."""
    payload = loader.policy_payload(record)
    return PolicyObservation(payload['instruction'], tuple(payload['rgb']),
                             tuple(a.lower() for a in payload['executed_actions']))


def policy_arguments(observation, memory, *, memory_instruction):
    if type(observation) is not PolicyObservation:
        raise TypeError('POLICY_OBSERVATION_ONLY')
    if memory_instruction != observation.instruction:
        raise ValueError('TASK_CHANGE_REQUIRES_PREFIX_RECOMPUTATION')
    return dict(instruction=observation.instruction, rgb=observation.rgb,
                executed=observation.executed, memory=memory)


def continuation_query(compiler, query):
    """Training reader only; local category indices remain bound to vocabulary."""
    if set(query) != {'query_schema_version', 'coordinate_frame', 'sequence'}:
        raise ValueError('QUERY_FIREWALL: explicit semantic projection required')
    encoded = compiler.encode_query(query)
    if set(encoded) != {'token_ids', 'vocabulary', 'encoding_schema'}:
        raise ValueError('UNBOUND_QUERY_ENCODING')
    return encoded


def family_aux_weights(families):
    """Equal weight per family with known labels; no unknown-to-negative coercion."""
    counts = [sum(c.mask for c in cells) for cells in families]
    active = sum(n > 0 for n in counts)
    if active == 0:
        raise ValueError('NO_VALID_AUXILIARY_FAMILY')
    return [[c.mask / (active * n) if n else 0. for c in cells]
            for cells, n in zip(families, counts)]
