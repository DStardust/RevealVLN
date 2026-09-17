"""Real short AF_UNIX bind and torch DataLoader fd-transfer before GPU admission."""
import json,os,socket,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;LINE=HERE.parents[1]
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ['TMPDIR']==str(LINE/'.t9')
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    assert not torch.cuda.is_initialized()
    assert tempfile.gettempdir()==str(LINE/'.t9')
    address=str(LINE/'.t9'/'listener_probe')
    assert len(address.encode())<108 and not Path(address).exists()
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:sock.bind(address);sock.listen(1)
    finally:sock.close()
    # Remove only this exact test-created socket, not any directory or user data.
    assert Path(address).is_socket();Path(address).unlink()
    x=torch.arange(64*12,dtype=torch.float32).reshape(64,12);y=torch.arange(64)
    loader=DataLoader(TensorDataset(x,y),batch_size=4,num_workers=2,prefetch_factor=2,timeout=30)
    seen=[]
    for xx,yy in loader:
        assert torch.equal(xx,x[yy]);seen.extend(yy.tolist())
    assert seen==list(range(64))
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS',unix=time.time(),actual_unix_socket_bind=True,torch_loader_workers=2,
      cpu_tensor_rows_transferred=64,all_rows_exact=True,temporary_directory=tempfile.gettempdir(),
      max_expected_worker_socket_path_bytes=len(str(LINE/'.t9').encode())+len('/pymp-abcdefgh/listener-abcdefgh'),
      gpu_actions=0,optimizer_steps=0,only_removed_object='exact test-created listener_probe socket')
    with (HERE/'IPC_TEST_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':main()
