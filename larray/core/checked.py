from abc import ABCMeta
import warnings

import numpy as np

from typing import Type, Any, Dict, Set, no_type_check, Optional, Annotated

from larray.core.axis import AxisCollection
from larray.core.array import Array, full
from larray.core.session import Session


class NotLoaded:
    pass


try:
    import pydantic
except ImportError:
    pydantic = None


# the not implemented versions of Checked* classes must be in the beginning of
# the module otherwise PyCharm do not provide auto-completion for methods of
# CheckedSession (imported from Session)
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
    from pydantic import ConfigDict, BeforeValidator, ValidationInfo, TypeAdapter, ValidationError
    from pydantic_core import PydanticUndefined

    def CheckedArray(axes: AxisCollection, dtype: np.dtype = float) -> Type[Array]:
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
        expected_axes = axes

        dtype = np.dtype(dtype)

        def validate_array(value: Any, info: ValidationInfo) -> Array:
            name = info.context.get("name", "<unknown>")
            if not (isinstance(value, Array) or np.isscalar(value)):
                raise TypeError(f"Expected object of type '{Array.__name__}' or a scalar for "
                                f"the variable '{name}' but got object of type '{type(value).__name__}'")

            # check axes
            if isinstance(value, Array):
                error_msg = f"Array '{name}' was declared with axes {expected_axes} but got array " \
                            f"with axes {value.axes}"
                # check for extra axes
                extra_axes = value.axes - expected_axes
                if extra_axes:
                    raise ValueError(f"{error_msg} (unexpected {extra_axes} "
                                     f"{'axes' if len(extra_axes) > 1 else 'axis'})")
                # check compatible axes
                try:
                    expected_axes.check_compatible(value.axes)
                except ValueError as error:
                    error_msg = str(error).replace("incompatible axes",
                                                   f"Incompatible axis for array '{name}'")
                    raise ValueError(error_msg)
                # broadcast + transpose if needed
                value = value.expand(expected_axes)
                # check dtype
                if value.dtype != dtype:
                    value = value.astype(dtype)
                return value
            else:
                return full(axes=expected_axes, fill_value=value, dtype=dtype)

        return Annotated[Array, BeforeValidator(validate_array)]


    class AbstractCheckedSession:
        pass


    # Simplified version of the ModelMetaclass class from pydantic:
    # https://github.com/pydantic/pydantic/blob/v2.12.0/pydantic/_internal/_model_construction.py
    class ModelMetaclass(ABCMeta):
        @no_type_check  # noqa C901
        def __new__(mcs, cls_name: str, bases: tuple[type[Any], ...], namespace: dict[str, Any], **kwargs: Any):
            from pydantic._internal._config import ConfigWrapper
            from pydantic._internal._decorators import DecoratorInfos
            from pydantic._internal._namespace_utils import NsResolver
            from pydantic._internal._fields import is_valid_field_name
            from pydantic._internal._model_construction import (inspect_namespace, set_model_fields,
                                                                complete_model_class, set_default_hash_func)

            raw_annotations = namespace.get('__annotations__', {})

            # tries to infer types for variables without type hints
            keys_to_infer_type = [key for key in namespace.keys()
                                  if key not in raw_annotations]
            keys_to_infer_type = [key for key in keys_to_infer_type
                                  if is_valid_field_name(key)]
            keys_to_infer_type = [key for key in keys_to_infer_type
                                  if key not in {'model_config', 'dict', 'build'}]
            keys_to_infer_type = [key for key in keys_to_infer_type
                                  if not callable(namespace[key])]
            for key in keys_to_infer_type:
                value = namespace[key]
                raw_annotations[key] = type(value)

            base_field_names, class_vars, base_private_attributes = mcs._collect_bases_data(bases)

            config_wrapper = ConfigWrapper.for_model(bases, namespace, raw_annotations, kwargs)
            namespace['model_config'] = config_wrapper.config_dict
            private_attributes = inspect_namespace(namespace, raw_annotations, config_wrapper.ignored_types,
                                                   class_vars, base_field_names)

            namespace['__class_vars__'] = class_vars
            namespace['__private_attributes__'] = {**base_private_attributes, **private_attributes}

            cls = super().__new__(mcs, cls_name, bases, namespace, **kwargs)

            cls.__pydantic_decorators__ = DecoratorInfos.build(cls)
            cls.__pydantic_decorators__.update_from_config(config_wrapper)

            cls.__pydantic_generic_metadata__ = {'origin': None, 'args': (), 'parameters': None}
            cls.__pydantic_root_model__ = False
            cls.__pydantic_complete__ = False

            # create a copy of raw_annotations since cls.__annotations__ points to it and
            # cls.__annotations__ must not be polluted before calling set_model_fields() later
            cls.__fields_annotations__ = {k: v for k, v in raw_annotations.items()}
            for base in reversed(bases):
                if issubclass(base, AbstractCheckedSession) and base != AbstractCheckedSession:
                    base_fields_annotations = getattr(base, '__fields_annotations__', {})
                    for k, v in base_fields_annotations.items():
                        if k not in cls.__fields_annotations__:
                            cls.__fields_annotations__[k] = v

            # preserve `__set_name__` protocol defined in https://peps.python.org/pep-0487
            # for attributes not in `namespace` (e.g. private attributes)
            for name, obj in private_attributes.items():
                obj.__set_name__(cls, name)

            ns_resolver = NsResolver()
            set_model_fields(cls, config_wrapper=config_wrapper, ns_resolver=ns_resolver)
            complete_model_class(cls, config_wrapper, ns_resolver, raise_errors=False, call_on_complete_hook=False)

            if config_wrapper.frozen and '__hash__' not in namespace:
                set_default_hash_func(cls, bases)

            return cls

        @staticmethod
        def _collect_bases_data(bases: tuple[type[Any], ...]) -> tuple[set[str], set[str], dict[str, Any]]:
            from pydantic.fields import ModelPrivateAttr

            field_names: set[str] = set()
            class_vars: set[str] = set()
            private_attributes: dict[str, ModelPrivateAttr] = {}
            for base in bases:
                if issubclass(base, AbstractCheckedSession) and base is not AbstractCheckedSession:
                    # model_fields might not be defined yet in the case of generics, so we use getattr here:
                    field_names.update(getattr(base, '__pydantic_fields__', {}).keys())
                    class_vars.update(base.__class_vars__)
                    private_attributes.update(base.__private_attributes__)
            return field_names, class_vars, private_attributes

        @property
        def __pydantic_fields_complete__(self) -> bool:
            """Whether the fields where successfully collected (i.e. type hints were successfully resolves).

            This is a private attribute, not meant to be used outside Pydantic.
            """
            if '__pydantic_fields__' not in self.__dict__:
                return False

            field_infos = self.__pydantic_fields__
            return all(field_info._complete for field_info in field_infos.values())

        def __dir__(self) -> list[str]:
            attributes = list(super().__dir__())
            if '__fields__' in attributes:
                attributes.remove('__fields__')
            return attributes


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
        TypeError: Error while assigning value to variable 'birth_rate':
        Input should be an instance of Array. Got input value of type 'list'.
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
        ValueError: Error while assigning value to variable 'mortality_rate':
        Array 'mortality_rate' was declared with axes {age, gender} but got array with axes
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
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True, extra='allow',
                                  validate_assignment=True, frozen=False)

        def __init__(self, *args, meta=None, **kwargs):
            Session.__init__(self, meta=meta)

            # create an intermediate Session object to not call the __setattr__
            # and __setitem__ overridden in the present class and in case a filepath
            # is given as only argument
            # TODO: refactor Session.load() to use a private function which returns the handler directly
            # so that we can get the items out of it and avoid this
            input_data = dict(Session(*args, **kwargs))

            # --- declared variables
            for name, field in self.__pydantic_fields__.items():
                value = input_data.pop(name, NotLoaded())

                if isinstance(value, NotLoaded):
                    if field.default is PydanticUndefined:
                        warnings.warn(f"No value passed for the declared variable '{name}'", stacklevel=2)
                        self.__setattr__(name, value, skip_frozen=True, skip_validation=True)
                    else:
                        self.__setattr__(name, field.default, skip_frozen=True)
                else:
                    self.__setattr__(name, value, skip_frozen=True)

            # --- undeclared variables
            for name, value in input_data.items():
                self.__setattr__(name, value, skip_frozen=True, stacklevel=2)

        # code of the method below has been partly borrowed from pydantic.BaseModel.__setattr__()
        def _check_key_value(self, name: str, value: Any, skip_frozen: bool, skip_validation: bool,
                             stacklevel: int) -> Any:
            config = self.model_config
            if not config['extra'] and name not in self.__pydantic_fields__:
                raise ValueError(f"Variable '{name}' is not declared in '{self.__class__.__name__}'. "
                                 f"Adding undeclared variables is forbidden. "
                                 f"List of declared variables is: {list(self.__pydantic_fields__.keys())}.")
            if not skip_frozen and config['frozen']:
                raise TypeError(f"Cannot change the value of the variable '{name}' since '{self.__class__.__name__}' "
                                f"is immutable and does not support item assignment")
            if name in self.__pydantic_fields__:
                if not skip_validation:
                    try:
                        field_type = self.__fields_annotations__.get(name, None)
                        if field_type is None:
                            return value
                        # see https://docs.pydantic.dev/2.12/concepts/types/#custom-types
                        # for more details about TypeAdapter
                        adapter = TypeAdapter(field_type, config=self.model_config)
                        value = adapter.validate_python(value, context={'name': name})
                    except ValidationError as e:
                        error = e.errors()[0]
                        msg = f"Error while assigning value to variable '{name}':\n"
                        if error['type'] == 'is_instance_of':
                            msg += error['msg']
                            msg += f". Got input value of type '{type(value).__name__}'."
                            raise TypeError(msg)
                        if error['type'] == 'value_error':
                            msg += error['ctx']['error'].args[0]
                        else:
                            msg += error['msg']
                        raise ValueError(msg)

            else:
                warnings.warn(f"'{name}' is not declared in '{self.__class__.__name__}'",
                              stacklevel=stacklevel + 1)
            return value

        def _update_from_iterable(self, it):
            for k, v in it:
                self.__setitem__(k, v, stacklevel=3)

        def __setitem__(self, key, value, skip_frozen=False, skip_validation=False, stacklevel=1):
            if key != 'meta':
                value = self._check_key_value(key, value, skip_frozen, skip_validation, stacklevel=stacklevel + 1)
                # we need to keep the attribute in sync
                object.__setattr__(self, key, value)
                self._objects[key] = value

        def __setattr__(self, key, value, skip_frozen=False, skip_validation=False, stacklevel=1):
            if key != 'meta':
                value = self._check_key_value(key, value, skip_frozen, skip_validation, stacklevel=stacklevel + 1)
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
        model_config = ConfigDict(frozen=True)
