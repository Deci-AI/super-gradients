import asyncio


def async_test_runner(async_io_co_routine):
    """
    Used as a decorator to run asynchronous testing
        :param async_io_co_routine:
        :return: Wrapped function
    """

    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(async_io_co_routine(*args, **kwargs))

    return wrapper
