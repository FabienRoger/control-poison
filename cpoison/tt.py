from attrs import define


class WithInfo:
    def info(self):
        return repr(self)


@define
class A(WithInfo):
    x: int = 2


print(A().info())
