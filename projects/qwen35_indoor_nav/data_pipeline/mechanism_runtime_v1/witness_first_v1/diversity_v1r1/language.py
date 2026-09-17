"""Finite controlled language with inverse parsing, not unconstrained paraphrases."""
import copy
from fractions import Fraction
import re

from common import WF, load

base = load('diversity_frozen_language', WF / 'language_realization_v1/verbalizer.py')
VERSION = 'q35n.controlled_language.v2'
TEMPLATES = (
    'First see {a} in two consecutive observations. Then see {b} in two consecutive observations and stop immediately.',
    'See {a} in two consecutive observations before seeing {b} in two consecutive observations; stop immediately afterward.',
    'Begin by seeing {a} in two consecutive observations. Next, see {b} in two consecutive observations and stop immediately.',
    'After seeing {a} in two consecutive observations, see {b} in two consecutive observations, then stop immediately.',
    'Complete these steps in order: see {a} in two consecutive observations; see {b} in two consecutive observations; stop immediately.',
)


def ast(roles, task):
    assert set(task) == {'anchor', 'terminal', 'instruction'}
    assert task['anchor'] != task['terminal']
    for r in roles.values():
        base.role_phrase(r)
    return dict(program='observable_see2_then_stop.v4',
                anchor=copy.deepcopy(roles[task['anchor']]),
                terminal=copy.deepcopy(roles[task['terminal']]),
                consecutive_observations=2, same_instance=True,
                temporal_relation='anchor_before_terminal_witness_then_immediate_stop')


def parse(instruction, roles):
    phrases = {}
    for key, role in roles.items():
        phrase = base.role_phrase(role)
        if phrase in phrases:
            raise ValueError('AMBIGUOUS_ROLE_PHRASE')
        phrases[phrase] = key
    matches = []
    for template in TEMPLATES:
        pattern = re.escape(template).replace(r'\{a\}', '(?P<a>.+?)').replace(r'\{b\}', '(?P<b>.+?)')
        match = re.fullmatch(pattern, instruction)
        if match and match['a'] in phrases and match['b'] in phrases:
            task = dict(anchor=phrases[match['a']], terminal=phrases[match['b']], instruction=instruction)
            matches.append(ast(roles, task))
    if len(matches) != 1:
        raise ValueError('UNREGISTERED_OR_AMBIGUOUS_INSTRUCTION')
    return matches[0]


def realize(roles, tasks, variant):
    if type(variant) is not int or not 0 <= variant < len(TEMPLATES):
        raise ValueError('VARIANT')
    result = copy.deepcopy(tasks)
    for index, (key, task) in enumerate(result.items()):
        chosen = (variant + index) % len(TEMPLATES)
        task['instruction'] = TEMPLATES[chosen].format(
            a=base.role_phrase(roles[task['anchor']]), b=base.role_phrase(roles[task['terminal']]))
        if parse(task['instruction'], roles) != ast(roles, tasks[key]):
            raise ValueError('AST_MISMATCH')
    return result


def aliases(roles, tasks, source_family):
    result = []
    for variant in range(len(TEMPLATES)):
        result.append(dict(variant=variant, tasks=realize(roles, tasks, variant),
                           physical_family_group=source_family,
                           relative_family_weight=str(Fraction(1, len(TEMPLATES))),
                           training_admission=False, physical_replay_required=True))
    return result
