import time, threading, pytest
from core.cache import LRUCache
def test_basic_set_get():
    c = LRUCache(maxsize=3, ttl=60)
    c.set('a', 1); assert c.get('a') == 1

def test_lru_eviction():
    c = LRUCache(maxsize=2, ttl=60)
    c.set('a',1); c.set('b',2); c.set('c',3)
    assert c.get('a') is None
    assert c.get('b') == 2

def test_lru_order_update():
    c = LRUCache(maxsize=2, ttl=60)
    c.set('a',1); c.set('b',2)
    c.get('a')          # a becomes most-recent
    c.set('c',3)        # b should be evicted
    assert c.get('b') is None
    assert c.get('a') == 1

def test_ttl_expiry():
    c = LRUCache(maxsize=10, ttl=0.05)
    c.set('x', 99)
    time.sleep(0.1)
    assert c.get('x') is None

def test_delete():
    c = LRUCache(); c.set('k','v'); c.delete('k')
    assert c.get('k') is None

def test_thread_safety():
    c = LRUCache(maxsize=100, ttl=60)
    def writer(n):
        for i in range(50): c.set(f'{n}-{i}', i)
    threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(c) <= 100