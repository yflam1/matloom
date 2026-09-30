import sys


class AnyBase:
    @property
    def cname_(self) -> str:
        return self.__class__.__name__

    @classmethod
    def get_method_name(cls) -> str:
        return sys._getframe(1).f_code.co_name
