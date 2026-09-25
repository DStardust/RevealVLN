"""One shared decoder boundary for extraction, inference and live probes."""
class ActionBoundary:
    def __init__(self, header):
        self.header=list(header)
        if not self.header:raise ValueError('EMPTY_ASSISTANT_HEADER')
        self.start=None

    def offset(self, input_ids):
        ids=input_ids[0].tolist()
        if self.start is None:self.start=len(ids)
        generated=ids[self.start:]
        n=min(len(generated),len(self.header))
        if generated[:n]!=self.header[:n]:
            raise ValueError('UNEXPECTED_ASSISTANT_HEADER:'+str(generated))
        return None if len(generated)<len(self.header) else len(generated)-len(self.header)


def assistant_header(tokenizer):
    header=tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False)
    if tokenizer.decode(header)!='<|im_start|>assistant\n':
        raise ValueError('HEADER_TOKENIZATION_CHANGED')
    return header

