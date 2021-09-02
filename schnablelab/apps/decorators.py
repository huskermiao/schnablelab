#This formula is a good boilerplate template for building more complex decorators.

import functools

def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator

# function based decorator with arguments
def decorator(arg1, arg2):
    def real_decorator(function):
        def wrapper(*args, **kwargs):
            print("Congratulations.  You decorated a function that does something with %s and %s" % (arg1, arg2))
            value = function(*args, **kwargs)
            # do something after
            return value
        return wrapper
    return real_decorator

# class based decorator
class ClassBasedDecorator(object):
    def __init__(self, func_to_decorate):
        print("INIT ClassBasedDecorator")
        self.func_to_decorate = func_to_decorate

    def __call__(self, *args, **kwargs):
        print("CALL ClassBasedDecorator")
        return self.func_to_decorate(*args, **kwargs)

# class based decorator with arguments
class ClassBasedDecoratorWithParams(object):
    def __init__(self, arg1, arg2):
        print("INIT ClassBasedDecoratorWithParams")
        print(arg1)
        print(arg2)

    def __call__(self, func_to_decorate, *args, **kwargs):
        print("CALL ClassBasedDecoratorWithParams")
        def new_func(*args, **kwargs):
            print("Function has been decorated.  Congratulations.")
            return func_to_decorate(*args, **kwargs)
        return new_func

#############
import time

def timer(func):
    '''
    If you want to do more precise measurements of code, 
    you should instead consider the timeit module in the standard library. 
    It temporarily disables garbage collection and runs multiple trials to strip out noise from quick function calls.
    '''
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f'Finished {func.__name__!r} in {run_time:.4f} secs')
        return value
    return wrapper_timer

def debug(func):
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = ['%s=%s'%(k,v) for k, v in kwargs.items()]
        signatures = ', '.join(args_repr, kwargs_repr)
        print('calling %s(%s)'%(func.__name__, signatures))
        value = func(*args, **kwargs_repr)
        print('%s returned %s'%(func.__name__, value))
        return value
    return wrapper_debug
#It’s more powerful when applied to small convenience functions that you don’t call directly yourself.
'''
math.factorial = debug(math.factorial)
def approximate_e(terms=18):
    return sum(1 / math.factorial(n) for n in range(terms))
'''

class SlowDown():
    '''
    the most common use case is that you want to rate-limit a function that continuously checks whether a resource—like a web page—has changed. 
    '''
    def __init__(self, sec=1):
        self.sec = sec
    
    def __call__(self, func, *args, **kwargs):
        @functools.wraps(func)
        def wrapper_slowdown(*args, **kwargs):
            time.sleep(self.sec)
            print('%s seconds went away...'%self.sec)
            value = func(*args, **kwargs)
            return value
        return wrapper_slowdown

#simply stores a reference to the decorated function in the global PLUGINS dict.
PLUGINS = dict()
def register(func):
    PLUGINS[func.__name__] = func
    return func
#Using the @register decorator, you can create your own curated list of interesting variables, 
#effectively hand-picking some functions from globals().


def set_unit(unit):
    """Register a unit on a function"""
    def decorator_set_unit(func):
        func.unit = unit
        return func
    return decorator_set_unit

'''
# ensure that the keys are part of the request
def validate_json(*expected_args):                  # 1
    def decorator_validate_json(func):
        @functools.wraps(func)
        def wrapper_validate_json(*args, **kwargs):
            json_object = request.get_json()
            for expected_arg in expected_args:      # 2
                if expected_arg not in json_object:
                    abort(400)
            return func(*args, **kwargs)
        return wrapper_validate_json
    return decorator_validate_json
'''