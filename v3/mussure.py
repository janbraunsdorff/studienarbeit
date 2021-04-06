import time 

def benchmark(func, *args):
    ret = ()
    start = time.time()
    ret = func(*args)
    end = time.time()
    
    print("[Time] ", time.strftime('%H:%M:%S', time.gmtime(end - start)), end=" ")
    return ret