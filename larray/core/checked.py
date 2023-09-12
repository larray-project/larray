from abc import ABCMeta
from copy import deepcopy
import warnings

import numpy as np

from typing import TYPE_CHECKING, Type, Any, Dict, Set, List, no_type_check, Optional

from larray.core.axis import AxisCollection
from larray.core.array import Array, full
from larray.core.session import Session


class NotLoaded:
    pass


try:
    import pydantic
except ImportError:
    pydantic = None

#  moved the not implemented versions of Checked* classes in the beginning of the module
#  otherwise PyCharm do not provide auto-completion for methods of CheckedSession
#  (imported from Session)
if not pydantic:
    def CheckedArray(axes: AxisCollection, dtype: np.dtype = float) -> Type[Array]:
        raise NotImplementedError("CheckedArray cannot be used because pydantic is not installed")

    class CheckedSession:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("CheckedSession class cannot be instantiated "
                                      "because pydantic is not installed")

    class CheckedParameters:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("CheckedParameters class cannot be instantiated "
                                      "because pydantic is not installed")
else:
    from pydantic.utils import Obj, IMMUTABLE_NON_COLLECTIONS_TYPES, BUILTIN_COLLECTIONS
    from pydantic.fields import ModelField
    from pydantic.class_validators import Validator
    from pydantic.main import BaseConfig

    # the implementation of the class below is inspired by the 'ConstrainedBytes' class
    # from the types.py module of the 'pydantic' library
    class CheckedArrayImpl(Array):
        expected_axes: AxisCollection
        dtype: np.dtype = np.dtype(float)

        # see https://pydantic-docs.helpmanual.io/usage/types/#classes-with-__get_validators__
        @classmethod
        def __get_validators__(cls):
            # one or more validators may be yielded which will be called in the
            # order to validate the input, each validator will receive as an input
            # the value returned from the previous validator
            yield cls.validate

        @classmethod
        def validate(cls, value, field: ModelField) -> Array:
            if not (isinstance(value, Array) or np.isscalar(value)):
                raise TypeError(f"Expected object of type '{Array.__name__}' or a scalar for "
                                f"the variable '{field.name}' but got object of type '{type(value).__name__}'")

            # check axes
            if isinstance(value, Array):
                error_msg = f"Array '{field.name}' was declared with axes {cls.expected_axes} but got array " \
                            f"with axes {value.axes}"
                # check for extra axes
                extra_axes = value.axes - cls.expected_axes
                if extra_axes:
                    raise ValueError(f"{error_msg} (unexpected {extra_axes} "
                                     f"{'axes' if len(extra_axes) > 1 else 'axis'})")
                # check compatible axes
                try:
                    cls.expected_axes.check_compatible(value.axes)
                except ValueError as error:
                    error_msg = str(error).replace("incompatible axes", f"Incompatible axis for array '{field.name}'")
                    raise ValueError(error_msg)
                # broadcast + transpose if needed
                value = value.expand(cls.expected_axes)
                # check dtype
                if value.dtype != cls.dtype:
                    value = value.astype(cls.dtype)
                return value
            else:
                return full(axes=cls.expected_axes, fill_value=value, dtype=cls.dtype)

    # the implementation of the function below is inspired by the 'conbytes' function
    # from the types.py module of the 'pydantic' library

    def CheckedArray(axes: AxisCollection, dtype: np.dtype = float) -> Type[Array]:
        # XXX: for a very weird reason I don't know, I have to put the fake import below
        #      to get autocompletion from PyCharm
        from larray.core.checked import CheckedArrayImpl
        """
        Represents a constrained array. It is intended to only be used along with :py:class:`CheckedSession`.

        Its axes are assumed to be "frozen", meaning they are constant all along the execution of the program.
        A constraint on the dtype of the data can be also specified.

        Parameters
        ----------
        axes: AxisCollection
            Axes of the checked array.
        dtype: data-type, optional
            Data-type for the checked array. Defaults to float.

        Returns
        -------
        Array
            Constrained array.
        """
        if axes is not None and not isinstance(axes, AxisCollection):
            axes = AxisCollection(axes)
        _dtype = np.dtype(dtype)

        class ArrayDefValue(CheckedArrayImpl):
            expected_axes = axes
            dtype = _dtype

        return ArrayDefValue

    class AbstractCheckedSession:
        pass

    # the original version of smart_deepcopy() (from pydantic) crashes when obj is of type of
    # np.ndarray or Array because the second if is written as:
    # elif not obj and obj_type in BUILTIN_COLLECTIONS:
    # which throws the error:
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    # see https://github.com/samuelcolvin/pydantic/issues/2923
    def smart_deepcopy(obj: Obj) -> Obj:
        """
        Return type as is for immutable built-in types
        Use obj.copy() for built-in empty collections
        Use copy.deepcopy() for non-empty collections and unknown objects.
        """
        obj_type = obj.__class__
        if obj_type in IMMUTABLE_NON_COLLECTIONS_TYPES:
            return obj  # fastest case: obj is immutable and not collection therefore will not be copied anyway
        elif obj_type in BUILTIN_COLLECTIONS and not obj:
            # faster way for empty collections, no need to copy its members
            return obj if obj_type is tuple else obj.copy()  # type: ignore  # tuple doesn't have copy method
        return deepcopy(obj)  # slowest way when we actually might need a deepcopy

    class LModelField(ModelField):
        def get_default(self) -> Any:
            return smart_deepcopy(self.default) if self.default_factory is None else self.default_factory()

    # Simplified version of the ModelMetaclass class from pydantic:
    # https://github.com/samuelcolvin/pydantic/blob/master/pydantic/main.py

    class ModelMetaclass(ABCMeta):
        @no_type_check  # noqa C901
        def __new__(mcs, name, bases, namespace, **kwargs):
            from pydantic.fields import Undefined
            from pydantic.class_validators import extract_validators, inherit_validators
            from pydantic.types import PyObject
            from pydantic.typing import is_classvar, resolve_annotations
            from pydantic.utils import lenient_issubclass, validate_field_name
            from pydantic.main import inherit_config, prepare_config, UNTOUCHED_TYPES

            fields: Dict[str, ModelField] = {}
            config = BaseConfig
            validators: Dict[str, List[Validator]] = {}

            for base in reversed(bases):
                if issubclass(base, AbstractCheckedSession) and base != AbstractCheckedSession:
                    config = inherit_config(base.__config__, config)
                    fields.update(deepcopy(base.__fields__))
                    validators = inherit_validators(base.__validators__, validators)

            config = inherit_config(namespace.get('Config'), config)
            validators = inherit_validators(extract_validators(namespace), validators)

            # update fields inherited from base classes
            for field in fields.values():
                field.set_config(config)
                extra_validators = validators.get(field.name, [])
                if extra_validators:
                    field.class_validators.update(extra_validators)
                    # re-run prepare to add extra validators
                    field.populate_validators()

            prepare_config(config, name)

            # extract and build fields
            class_vars = set()
            if (namespace.get('__module__'), namespace.get('__qualname__')) != \
                    ('larray.core.checked', 'CheckedSession'):
                untouched_types = UNTOUCHED_TYPES + config.keep_untouched

                # annotation only fields need to come first in fields
                annotations = resolve_annotations(namespace.get('__annotations__', {}),
                                                  namespace.get('__module__', None))
                for ann_name, ann_type in annotations.items():
                    if is_classvar(ann_type):
                        class_vars.add(ann_name)
                    elif not ann_name.startswith('_'):
                        validate_field_name(bases, ann_name)
                        value = namespace.get(ann_name, Undefined)
                        if (isinstance(value, untouched_types) and ann_type != PyObject
                                and not lenient_issubclass(getattr(ann_type, '__origin__', None), Type)):
                            continue
                        fields[ann_name] = LModelField.infer(name=ann_name, value=value, annotation=ann_type,
                                                             class_validators=validators.get(ann_name, []),
                                                             config=config)

                for var_name, value in namespace.items():
                    # 'var_name not in annotations' because namespace.items() contains annotated fields
                    # with default values
                    # 'var_name not in class_vars' to avoid to update a field if it was redeclared (by mistake)
                    if (var_name not in annotations and not var_name.startswith('_')
                            and not isinstance(value, untouched_types) and var_name not in class_vars):
                        validate_field_name(bases, var_name)
                        # since pydantic 1.6, ModelField.infer() fails to infer the type (it is set to None)
                        annotation = type(value)
                        inferred = LModelField.infer(name=var_name, value=value, annotation=annotation,
                                                     class_validators=validators.get(var_name, []), config=config)
                        if var_name in fields and inferred.type_ != fields[var_name].type_:
                            raise TypeError(f'The type of {name}.{var_name} differs from the new default value; '
                                            f'if you wish to change the type of this field, please use a type '
                                            f'annotation')
                        fields[var_name] = inferred

            new_namespace = {
                '__config__': config,
                '__fields__': fields,
                '__field_defaults__': {n: f.default for n, f in fields.items() if not f.required},
                '__validators__': validators,
                **{n: v for n, v in namespace.items() if n not in fields},
            }
            return super().__new__(mcs, name, bases, new_namespace, **kwargs)

    class CheckedSession(Session, AbstractCheckedSession, metaclass=ModelMetaclass):
        """
        Class intended to be inherited by user defined classes in which the variables of a model are declared.
        Each declared variable is constrained by a type defined explicitly or deduced from the given default value
        (see examples below).
        All classes inheriting from `CheckedSession` will have access to all methods of the :py:class:`Session` class.

        The special :py:obj:`CheckedArray` type represents an Array object with fixed axes and/or dtype.
        This prevents users from modifying the dimensions (and labels) and/or the dtype of an array by mistake
        and make sure that the definition of an array remains always valid in the model.

        By declaring variables, users will speed up the development of their models using the auto-completion
        (the feature in which development tools like PyCharm try to predict the variable or function a user intends
        to enter after only a few characters have been typed).

        As for normal Session objects, it is still possible to add undeclared variables to instances of
        classes inheriting from `CheckedSession` but this must be done with caution.

        Parameters
        ----------
        *args : str or dict of {str: object} or iterable of tuples (str, object)
            Path to the file containing the session to load or
            list/tuple/dictionary containing couples (name, object).
        **kwargs : dict of {str: object}

            * Objects to add written as name=object
            * meta : list of pairs or dict or Metadata, optional
                Metadata (title, description, author, creation_date, ...) associated with the array.
                Keys must be strings. Values must be of type string, int, float, date, time or datetime.

        Warnings
        --------
        The :py:obj:`CheckedSession.filter()`, :py:obj:`CheckedSession.compact()`
        and :py:obj:`CheckedSession.apply()` methods return a simple Session object.
        The type of the declared variables (and the value for the declared constants) will
        no longer be checked.

        See Also
        --------
        Session, CheckedParameters

        Examples
        --------
        Content of file 'parameters.py'

        >>> from larray import *
        >>> FIRST_YEAR = 2020
        >>> LAST_YEAR = 2030
        >>> AGE = Axis('age=0..10')
        >>> GENDER = Axis('gender=male,female')
        >>> TIME = Axis(f'time={FIRST_YEAR}..{LAST_YEAR}')

        Content of file 'model.py'

        >>> class ModelVariables(CheckedSession):
        ...     # --- declare variables with defined types ---
        ...     # Their values will be defined at runtime but must match the specified type.
        ...     birth_rate: Array
        ...     births: Array
        ...     # --- declare variables with a default value ---
        ...     # The default value will be used to set the variable if no value is passed at instantiation (see below).
        ...     # Their type is deduced from their default value and cannot be changed at runtime.
        ...     target_age = AGE[:2] >> '0-2'
        ...     population = zeros((AGE, GENDER, TIME), dtype=int)
        ...     # --- declare checked arrays ---
        ...     # The checked arrays have axes assumed to be "frozen", meaning they are
        ...     # constant all along the execution of the program.
        ...     mortality_rate: CheckedArray((AGE, GENDER))
        ...     # For checked arrays, the default value can be given as a scalar.
        ...     # Optionally, a dtype can be also specified (defaults to float).
        ...     deaths: CheckedArray((AGE, GENDER, TIME), dtype=int) = 0

        >>> variant_name = "baseline"
        >>> # Instantiation --> create an instance of the ModelVariables class.
        >>> # Warning: All variables declared without a default value must be set.
        >>> m = ModelVariables(birth_rate = zeros((AGE, GENDER)),
        ...                    births = zeros((AGE, GENDER, TIME), dtype=int),
        ...                    mortality_rate = 0)

        >>> # ==== model ====
        >>> # In the definition of ModelVariables, the 'birth_rate' variable, has been declared as an Array object.
        >>> # This means that the 'birth_rate' variable will always remain of type Array.
        >>> # Any attempt to assign a non-Array value to 'birth_rate' will make the program to crash.
        >>> m.birth_rate = Array([0.045, 0.055], GENDER)    # OK
        >>> m.birth_rate = [0.045, 0.055]                   # Fails
        Traceback (most recent call last):
            ...
        pydantic.errors.ArbitraryTypeError: instance of Array expected
        >>> # However, the arrays 'birth_rate', 'births' and 'population' have not been declared as 'CheckedArray'.
        >>> # Thus, axes and dtype of these arrays are not protected, leading to potentially unexpected behavior
        >>> # of the model.
        >>> # example 1: Let's say we want to calculate the new births for the year 2025 and we assume that
        >>> # the birth rate only differ by gender.
        >>> # In the line below, we add an additional TIME axis to 'birth_rate' while it was initialized
        >>> # with the AGE and GENDER axes only
        >>> m.birth_rate = full((AGE, GENDER, TIME), fill_value=Array([0.045, 0.055], GENDER))
        >>> # here 'new_births' have the AGE, GENDER and TIME axes instead of the AGE and GENDER axes only
        >>> new_births = m.population['female', 2025] * m.birth_rate
        >>> print(new_births.info)
        11 x 2 x 11
         age [11]: 0 1 2 ... 8 9 10
         gender [2]: 'male' 'female'
         time [11]: 2020 2021 2022 ... 2028 2029 2030
        dtype: float64
        memory used: 1.89 Kb
        >>> # and the line below will crash
        >>> m.births[2025] = new_births         # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: Value {time} axis is not present in target subset {age, gender}.
        A value can only have the same axes or fewer axes than the subset being targeted
        >>> # now let's try to do the same for deaths and making the same mistake as for 'birth_rate'.
        >>> # The program will crash now at the first step instead of letting you go further
        >>> m.mortality_rate = full((AGE, GENDER, TIME), fill_value=sequence(AGE, inc=0.02))    \
        # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: Array 'mortality_rate' was declared with axes {age, gender} but got array with axes
        {age, gender, time} (unexpected {time} axis)

        >>> # example 2: let's say we want to calculate the new births for all years.
        >>> m.birth_rate = full((AGE, GENDER, TIME), fill_value=Array([0.045, 0.055], GENDER))
        >>> new_births = m.population['female'] * m.birth_rate
        >>> # here 'new_births' has the same axes as 'births' but is a float array instead of
        >>> # an integer array as 'births'.
        >>> # The line below will make the 'births' array become a float array while
        >>> # it was initialized as an integer array
        >>> m.births = new_births
        >>> print(m.births.info)
        11 x 11 x 2
         age [11]: 0 1 2 ... 8 9 10
         time [11]: 2020 2021 2022 ... 2028 2029 2030
         gender [2]: 'male' 'female'
        dtype: float64
        memory used: 1.89 Kb
        >>> # now let's try to do the same for deaths.
        >>> m.mortality_rate = full((AGE, GENDER), fill_value=sequence(AGE, inc=0.02))
        >>> # here the result of the multiplication of the 'population' array by the 'mortality_rate' array
        >>> # is automatically converted to an integer array
        >>> m.deaths = m.population * m.mortality_rate
        >>> print(m.deaths.info)                            # doctest: +SKIP
        11 x 2 x 11
         age [11]: 0 1 2 ... 8 9 10
         gender [2]: 'male' 'female'
         time [11]: 2020 2021 2022 ... 2028 2029 2030
        dtype: int32
        memory used: 968 bytes

        It is possible to add undeclared variables to a checked session
        but this will print a warning:

        >>> m.undeclared_var = 'my_value'                   # doctest: +SKIP
        UserWarning: 'undeclared_var' is not declared in 'ModelVariables'

        >>> # ==== output ====
        >>> # save all variables in an HDF5 file
        >>> m.save(f'{variant_name}.h5', display=True)      # doctest: +SKIP
        dumping birth_rate ... done
        dumping births ... done
        dumping mortality_rate ... done
        dumping deaths ... done
        dumping target_age ... done
        dumping population ... done
        dumping undeclared_var ... done
        """

        if TYPE_CHECKING:
            # populated by the metaclass, defined here to help IDEs only
            __fields__: Dict[str, ModelField] = {}
            __field_defaults__: Dict[str, Any] = {}
            __validators__: Dict[str, List[Validator]] = {}
            __config__: Type[BaseConfig] = BaseConfig

        class Config:
            # whether to allow arbitrary user types for fields (they are validated simply by checking
            # if the value is an instance of the type). If False, RuntimeError will be raised on model declaration.
            # (default: False)
            arbitrary_types_allowed = True
            # whether to validate field defaults
            validate_all = True
            # whether to ignore, allow, or forbid extra attributes during model initialization (and after).
            # Accepts the string values of 'ignore', 'allow', or 'forbid', or values of the Extra enum
            # (default: Extra.ignore)
            extra = 'allow'
            # whether to perform validation on assignment to attributes
            validate_assignment = True
            # whether models are faux-immutable, i.e. whether __setattr__ is allowed.
            # (default: True)
            allow_mutation = True

        # Warning: order of fields is not preserved.
        # As of v1.0 of pydantic all fields with annotations (whether annotation-only or with a default value)
        # will precede all fields without an annotation. Within their respective groups, fields remain in the
        # order they were defined.
        # See https://pydantic-docs.helpmanual.io/usage/models/#field-ordering
        def __init__(self, *args, meta=None, **kwargs):
            Session.__init__(self, meta=meta)

            # create an intermediate Session object to not call the __setattr__
            # and __setitem__ overridden in the present class and in case a filepath
            # is given as only argument
            # TODO: refactor Session.load() to use a private function which returns the handler directly
            # so that we can get the items out of it and avoid this
            input_data = dict(Session(*args, **kwargs))

            # --- declared variables
            for name, field in self.__fields__.items():
                value = input_data.pop(field.name, NotLoaded())

                if isinstance(value, NotLoaded):
                    if field.default is None:
                        warnings.warn(f"No value passed for the declared variable '{field.name}'", stacklevel=2)
                        self.__setattr__(name, value, skip_allow_mutation=True, skip_validation=True)
                    else:
                        self.__setattr__(name, field.default, skip_allow_mutation=True)
                else:
                    self.__setattr__(name, value, skip_allow_mutation=True)

            # --- undeclared variables
            for name, value in input_data.items():
                self.__setattr__(name, value, skip_allow_mutation=True, stacklevel=2)

        # code of the method below has been partly borrowed from pydantic.BaseModel.__setattr__()
        def _check_key_value(self, name: str, value: Any, skip_allow_mutation: bool, skip_validation: bool,
                             stacklevel: int) -> Any:
            config = self.__config__
            if not config.extra and name not in self.__fields__:
                raise ValueError(f"Variable '{name}' is not declared in '{self.__class__.__name__}'. "
                                 f"Adding undeclared variables is forbidden. "
                                 f"List of declared variables is: {list(self.__fields__.keys())}.")
            if not skip_allow_mutation and not config.allow_mutation:
                raise TypeError(f"Cannot change the value of the variable '{name}' since '{self.__class__.__name__}' "
                                f"is immutable and does not support item assignment")
            known_field = self.__fields__.get(name, None)
            if known_field:
                if not skip_validation:
                    value, error_ = known_field.validate(value, self.dict(exclude={name}), loc=name, cls=self.__class__)
                    if error_:
                        raise error_.exc
            else:
                warnings.warn(f"'{name}' is not declared in '{self.__class__.__name__}'", stacklevel=stacklevel + 1)
            return value

        def _update_from_iterable(self, it):
            for k, v in it:
                self.__setitem__(k, v, stacklevel=3)

        def __setitem__(self, key, value, skip_allow_mutation=False, skip_validation=False, stacklevel=1):
            if key != 'meta':
                value = self._check_key_value(key, value, skip_allow_mutation, skip_validation,
                                              stacklevel=stacklevel + 1)
                # we need to keep the attribute in sync
                object.__setattr__(self, key, value)
                self._objects[key] = value

        def __setattr__(self, key, value, skip_allow_mutation=False, skip_validation=False, stacklevel=1):
            if key != 'meta':
                value = self._check_key_value(key, value, skip_allow_mutation, skip_validation,
                                              stacklevel=stacklevel + 1)
                # we need to keep the attribute in sync
                object.__setattr__(self, key, value)
            Session.__setattr__(self, key, value)

        def __getstate__(self) -> Dict[str, Any]:
            return {'__dict__': self.__dict__}

        def __setstate__(self, state: Dict[str, Any]) -> None:
            object.__setattr__(self, '__dict__', state['__dict__'])

        def dict(self, exclude: Optional[Set[str]] = None) -> Dict[str, Any]:
            return {k: v for k, v in self.items() if k not in exclude}

    class CheckedParameters(CheckedSession):
        """
        Same as py:class:`CheckedSession` but declared variables cannot be modified after initialization.

        Parameters
        ----------
        *args : str or dict of {str: object} or iterable of tuples (str, object)
            Path to the file containing the session to load or
            list/tuple/dictionary containing couples (name, object).
        **kwargs : dict of {str: object}

            * Objects to add written as name=object
            * meta : list of pairs or dict or Metadata, optional
                Metadata (title, description, author, creation_date, ...) associated with the array.
                Keys must be strings. Values must be of type string, int, float, date, time or datetime.

        See Also
        --------
        CheckedSession

        Examples
        --------
        Content of file 'parameters.py'

        >>> from larray import *
        >>> class Parameters(CheckedParameters):
        ...     # --- declare variables with fixed values ---
        ...     # The given values can never be changed
        ...     FIRST_YEAR = 2020
        ...     LAST_YEAR = 2030
        ...     AGE = Axis('age=0..10')
        ...     GENDER = Axis('gender=male,female')
        ...     TIME = Axis(f'time={FIRST_YEAR}..{LAST_YEAR}')
        ...     # --- declare variables with defined types ---
        ...     # Their values must be defined at initialized and will be frozen after.
        ...     variant_name: str

        Content of file 'model.py'

        >>> # instantiation --> create an instance of the ModelVariables class
        >>> # all variables declared without value must be set
        >>> P = Parameters(variant_name='variant_1')
        >>> # once an instance is created, its variables can be accessed but not modified
        >>> P.variant_name
        'variant_1'
        >>> P.variant_name = 'new_variant'      # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        TypeError: Cannot change the value of the variable 'variant_name' since 'Parameters'
        is immutable and does not support item assignment
        """

        class Config:
            # whether models are faux-immutable, i.e. whether __setattr__ is allowed.
            # (default: True)
            allow_mutation = False
