"""Single-producer ordered CPU prefetch; bounded queue, visible exceptions, joined exit."""
import os
import queue
import threading

class OrderedPrefetch:
    def __init__(self, factory, capacity=2, timeout=90, name='cpu-prefetch'):
        if capacity != 2 or timeout <= 0:
            raise ValueError('PREFETCH_CONFIGURATION')
        self.factory = factory
        self.queue = queue.Queue(maxsize=capacity)
        self.timeout = timeout
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._produce, name=name, daemon=True)
        self.started = False
        self.finished = False
        self.max_queued = 0

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=.1)
                self.max_queued = max(self.max_queued, self.queue.qsize())
                return True
            except queue.Full:
                pass
        return False

    def _produce(self):
        iterator = None
        try:
            iterator = iter(self.factory())
            while not self.stop.is_set():
                try:
                    value = next(iterator)
                except StopIteration:
                    self._put(('done', None))
                    return
                if not self._put(('value', value)):
                    return
                del value
        except BaseException as exc:
            self._put(('error', exc))
        finally:
            if iterator is not None and hasattr(iterator, 'close'):
                iterator.close()

    def __iter__(self):
        if not self.started:
            if self.stop.is_set():
                raise RuntimeError('PREFETCH_CLOSED_BEFORE_START')
            self.started = True
            self.thread.start()
        return self

    def __next__(self):
        if self.finished:
            raise StopIteration
        if not self.started:
            iter(self)
        try:
            kind, value = self.queue.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError('CPU_PREFETCH_TIMEOUT')
        if kind == 'value':
            return value
        self.finished = True
        self.close()
        if kind == 'error':
            raise value
        if kind != 'done':
            raise RuntimeError('PREFETCH_MESSAGE')
        raise StopIteration

    def close(self):
        self.stop.set()
        if self.started:
            self.thread.join(timeout=self.timeout)
            if self.thread.is_alive():
                raise TimeoutError('CPU_PREFETCH_JOIN_TIMEOUT')
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.finished = True

    def is_alive(self):
        return self.thread.is_alive()

    def identity(self):
        return dict(pid=os.getpid(), name=self.thread.name, native_id=self.thread.native_id,
                    process_workers=0, queue_capacity=self.queue.maxsize)

