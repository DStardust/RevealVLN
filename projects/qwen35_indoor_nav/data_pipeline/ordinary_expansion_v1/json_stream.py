"""Bounded stdlib JSON streaming for official episode arrays / GT objects."""
import gzip
import json
class Stream:
    def __init__(self,handle):self.handle=handle;self.buffer='';self.done=False;self.decoder=json.JSONDecoder()
    def refill(self):
        data=self.handle.read(65536)
        if not data:self.done=True
        self.buffer+=data;assert len(self.buffer)<=16*1024**2,'SINGLE_JSON_VALUE_EXCEEDS_CPU_BOUND'
    def trim(self):
        self.buffer=self.buffer.lstrip()
        while not self.buffer and not self.done:self.refill();self.buffer=self.buffer.lstrip()
    def char(self,c):
        self.trim();assert self.buffer.startswith(c),(c,self.buffer[:20]);self.buffer=self.buffer[len(c):]
    def value(self):
        self.trim()
        while True:
            try:v,end=self.decoder.raw_decode(self.buffer);self.buffer=self.buffer[end:];return v
            except json.JSONDecodeError:
                if self.done:raise
                self.refill()
def episodes(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        s=Stream(f);s.char('{');assert s.value()=='episodes';s.char(':');s.char('[')
        s.trim()
        while not s.buffer.startswith(']'):
            yield s.value();s.trim()
            if s.buffer.startswith(','):s.char(',')
            else:break
        s.char(']');s.trim()
        while s.buffer.startswith(','):
            s.char(',');key=s.value();assert key=='instruction_vocab','UNREGISTERED_EPISODE_METADATA'
            s.char(':');s.value();s.trim()
        s.char('}');s.trim();assert not s.buffer
def entries(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        s=Stream(f);s.char('{');s.trim()
        while not s.buffer.startswith('}'):
            key=s.value();s.char(':');yield key,s.value();s.trim()
            if s.buffer.startswith(','):s.char(',')
            else:break
        s.char('}');s.trim();assert not s.buffer
