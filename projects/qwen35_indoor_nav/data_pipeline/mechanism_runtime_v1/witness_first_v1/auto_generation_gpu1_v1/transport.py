"""Fixed GPU1 UUID transport; original guard/clock/factory are untouched."""
import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('gpu1_transport_adapter',Path(__file__).resolve().parent/'adapter.py')
a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
m=a.module('transport.py')
WF=m.WF;BE=m.BE;QUEUE=m.QUEUE;private=m.private
sha=m.sha;load=m.load;base=m.base;CLOCK=m.CLOCK;CLOCK_SHA=m.CLOCK_SHA
check_inputs=m.check_inputs;worker_main=m.worker_main;run_main=m.run_main
allowed_batches=m.allowed_batches
