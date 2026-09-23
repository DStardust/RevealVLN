"""SSH CONNECT transport using proxyon's environment credentials, never argv."""
import base64
import os
import select
import socket
import sys
from urllib.parse import unquote, urlsplit


def main():
    host,port=sys.argv[1:]
    if (host,port) not in {('ssh.github.com','443'),('github.com','22')}:
        raise ValueError('Only GitHub SSH endpoints are allowed')
    proxy=urlsplit(os.environ.get('HTTPS_PROXY') or os.environ['https_proxy'])
    if proxy.scheme!='http' or not proxy.hostname or not proxy.port:
        raise ValueError('proxyon must provide an HTTP CONNECT proxy')
    headers=[f'CONNECT {host}:{port} HTTP/1.1',f'Host: {host}:{port}']
    if proxy.username is not None:
        credential=unquote(proxy.username)+':'+unquote(proxy.password or '')
        headers.append('Proxy-Authorization: Basic '+base64.b64encode(credential.encode()).decode())
    with socket.create_connection((proxy.hostname,proxy.port),timeout=20) as connection:
        connection.sendall(('\r\n'.join(headers)+'\r\n\r\n').encode())
        response=bytearray()
        while not response.endswith(b'\r\n\r\n'):
            byte=connection.recv(1)
            if not byte or len(response)>=65536:raise RuntimeError('Invalid CONNECT response')
            response.extend(byte)
        if bytes(response).split(b'\r\n',1)[0].split()[1]!=b'200':
            raise RuntimeError('Proxy rejected authenticated CONNECT')
        connection.settimeout(None)
        readers=[connection,sys.stdin.buffer]
        while connection in readers:
            for ready in select.select(readers,[],[])[0]:
                if ready is connection:
                    data=connection.recv(65536)
                    if not data:return
                    sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
                else:
                    data=os.read(sys.stdin.fileno(),65536)
                    if not data:
                        readers.remove(sys.stdin.buffer);connection.shutdown(socket.SHUT_WR)
                    else:connection.sendall(data)


if __name__=='__main__':main()
