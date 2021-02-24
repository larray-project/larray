import sys


class ExprNode(object):
    # method factory
    def _binop(opname):
        def opmethod(self, other):
            return BinaryOp(opname, self, other)

        opmethod.__name__ = f'__{opname}__'
        return opmethod

    __matmul__ = _binop('matmul')
    __ror__ = _binop('ror')
    __or__ = _binop('or')
    __rxor__ = _binop('rxor')
    __xor__ = _binop('xor')
    __rand__ = _binop('rand')
    __and__ = _binop('and')
    __rrshift__ = _binop('rrshift')
    __rshift__ = _binop('rshift')
    __rlshift__ = _binop('rlshift')
    __lshift__ = _binop('lshift')
    __rpow__ = _binop('rpow')
    __pow__ = _binop('pow')
    __rdivmod__ = _binop('rdivmod')
    __divmod__ = _binop('divmod')
    __rmod__ = _binop('rmod')
    __mod__ = _binop('mod')
    __rfloordiv__ = _binop('rfloordiv')
    __floordiv__ = _binop('floordiv')
    __rtruediv__ = _binop('rtruediv')
    __truediv__ = _binop('truediv')
    if sys.version < '3':
        __div__ = _binop('div')
        __rdiv__ = _binop('rdiv')
    __rmul__ = _binop('rmul')
    __mul__ = _binop('mul')
    __rsub__ = _binop('rsub')
    __sub__ = _binop('sub')
    __radd__ = _binop('radd')
    __add__ = _binop('add')
    __ge__ = _binop('ge')
    __gt__ = _binop('gt')
    __ne__ = _binop('ne')
    __eq__ = _binop('eq')
    __le__ = _binop('le')
    __lt__ = _binop('lt')

    def _unaryop(opname):
        def opmethod(self):
            return UnaryOp(opname, self)

        opmethod.__name__ = f'__{opname}__'
        return opmethod

    # unary ops do not need broadcasting so do not need to be overridden
    __neg__ = _unaryop('neg')
    __pos__ = _unaryop('pos')
    __abs__ = _unaryop('abs')
    __invert__ = _unaryop('invert')

    def evaluate(self, context):
        raise NotImplementedError()


def expr_eval(expr, context):
    # in the end it all comes down to AxisReference.evaluate(AxisCollection) which returns the Axis
    return expr.evaluate(context) if isinstance(expr, ExprNode) else expr


class BinaryOp(ExprNode):
    def __init__(self, op, expr1, expr2):
        self.opname = f'__{op}__'
        self.expr1 = expr1
        self.expr2 = expr2

    def evaluate(self, context):
        # TODO: implement eval via numexpr
        expr1 = expr_eval(self.expr1, context)
        expr2 = expr_eval(self.expr2, context)
        return getattr(expr1, self.opname)(expr2)


class UnaryOp(ExprNode):
    def __init__(self, op, expr):
        self.opname = f'__{op}__'
        self.expr = expr

    def evaluate(self, context):
        # TODO: implement eval via numexpr
        expr = expr_eval(self.expr, context)
        return getattr(expr, self.opname)()
