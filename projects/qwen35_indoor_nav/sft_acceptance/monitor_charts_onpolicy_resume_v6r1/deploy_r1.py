"""Protect persistent owned jobs, not holders intentionally draining during a lease."""
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=HERE/'deploy.py'
assert hashlib.sha256(P.read_bytes()).hexdigest()=='2f4f3a465a6142603715b0791111d5df88df305c3e5eefc0e003f699efef3a75'
text=P.read_text();start=text.index('def protected_pids():');end=text.index('def main():',start)
text=text[:start]+"""def protected_pids():
    owner=json.loads((LINE/'reviews/Q35N_ORDINARY_ONPOLICY_ADAPT_V6_TRANSPORT_R1/SEAL.json').read_text())['owned_training_lease']
    assert ident(owner['pid'])==owner
    return (3996270,112240,1563749,1353421,owner['pid'])


"""+text[end:]
exec(compile(text,__file__+':non-draining-protected-identities','exec'),globals())
