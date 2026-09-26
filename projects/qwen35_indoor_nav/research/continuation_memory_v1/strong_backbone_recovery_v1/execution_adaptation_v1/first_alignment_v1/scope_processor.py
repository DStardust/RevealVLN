"""Change only which generated action offsets receive the frozen residual."""
from runtime import ChunkActionProcessor


class ScopedProcessor(ChunkActionProcessor):
    def __init__(self, *args, scope, **kwargs):
        if scope not in ('ALL', 'FIRST'):
            raise ValueError('UNREGISTERED_INTERVENTION_SCOPE')
        super().__init__(*args, **kwargs)
        self.scope = scope

    def __call__(self, input_ids, scores):
        enabled = self.apply_residual
        self.apply_residual = enabled and (self.scope == 'ALL' or self.boundary.offset(input_ids) == 0)
        try:
            return super().__call__(input_ids, scores)
        finally:
            self.apply_residual = enabled
