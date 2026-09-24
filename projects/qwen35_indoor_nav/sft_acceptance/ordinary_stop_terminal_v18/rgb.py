"""Sensor validity is shape/type/transport, not scene texture content."""
import hashlib
def validate(rgb):
    if tuple(rgb.shape)!=(224,224,3):raise ValueError('RGB_SHAPE')
    if str(rgb.dtype)!='uint8':raise ValueError('RGB_DTYPE')
    return dict(shape=list(rgb.shape),dtype=str(rgb.dtype),minimum=int(rgb.min()),maximum=int(rgb.max()),std=float(rgb.std()),uniform=bool(rgb.min()==rgb.max()),sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
