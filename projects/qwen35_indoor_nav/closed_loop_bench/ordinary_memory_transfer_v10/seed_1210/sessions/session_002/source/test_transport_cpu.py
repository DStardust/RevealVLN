"""Real socket idle/EOF regression and exact simulator-semantics preservation."""
from pathlib import Path
import socket
import sys
import threading
import time
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import transport


class Tests(unittest.TestCase):
    def test_idle_peer_can_receive_later_request_and_eof(self):
        parent,child=socket.socketpair()
        child.settimeout(.01)
        service=transport.request_socket(child.detach())
        stream=service.makefile('rw')
        parent.settimeout(1)
        errors=[]
        def serve():
            try:
                self.assertEqual(stream.readline(),'reset\n')
                stream.write('done\n');stream.flush()
                self.assertEqual(stream.readline(),'')
            except BaseException as exc:errors.append(exc)
        worker=threading.Thread(target=serve,daemon=True)
        worker.start()
        try:
            time.sleep(.05)  # More than the inherited idle deadline.
            self.assertTrue(worker.is_alive())
            parent.sendall(b'reset\n')
            self.assertEqual(parent.recv(32),b'done\n')
            parent.shutdown(socket.SHUT_WR)
            worker.join(1)
            self.assertFalse(worker.is_alive())
            self.assertFalse(errors,errors)
        finally:
            parent.close();transport.close_stream(stream);service.close()

    def test_broken_stream_does_not_abort_remaining_cleanup(self):
        parent,child=socket.socketpair()
        stream=parent.makefile('wb')
        child.close()
        stream.write(b'pending')
        transport.close_stream(stream)
        self.assertTrue(stream.closed)
        parent.close()

    def test_executor_diff_is_transport_only(self):
        old=(HERE.parent/'ordinary_cycle_pair_recovery_v5/executor.py').read_text()
        expected=old.replace('HERE=Path(__file__).resolve().parent',
            "LOCAL=Path(__file__).resolve().parent\nsys.path.insert(0,str(LOCAL))\nfrom transport import request_socket\nHERE=LOCAL.parent/'ordinary_cycle_pair_recovery_v5'")
        expected=expected.replace('sock=socket.socket(fileno=int(sys.argv[1]));sock.settimeout(180)',
                                  'sock=request_socket(int(sys.argv[1]))')
        self.assertEqual((HERE/'executor.py').read_text(),expected)


if __name__=='__main__':unittest.main()
