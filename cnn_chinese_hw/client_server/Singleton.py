from _thread import allocate_lock

GLOBAL_LOCK = allocate_lock()
DInit = {}


class Singleton(object):
    def __new__(cls, *args, **kwargs):
        """
        Only ever store a single instance in memory (singleton pattern)
        """
        with GLOBAL_LOCK:
            if not hasattr(cls, '_state'):
                print("CREATING SINGLETON FOR:", cls)
                cls._state = {}
                init_run = [False]
                old_init = cls.__init__
                local_lock = allocate_lock() # CHECK SCOPE HERE!! =========================================

                def init(self, *args, **kw):
                    with local_lock: # Make sure initial init can't happen at same time!
                        if not init_run[0]:
                            old_init(self, *args, **kw)
                            init_run[0] = True

                cls.__init__ = init

        self = object.__new__(cls)
        self.__dict__ = cls._state
        return self


if __name__ == '__main__':
    class TestClass(Singleton):
        def __init__(self):
            print("Should only run once!")
            self.value = 'blah'

        def check(self):
            self.value
            #print(self.value)

    import _thread

    def fn():
        for x in range(500000):
            a = TestClass()
            b = TestClass()
            #assert(a == b)
            a.check()
            b.check()

    for x in range(500):
        _thread.start_new(fn, ())
    while 1: pass
