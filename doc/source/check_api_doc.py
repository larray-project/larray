import os
import inspect
import importlib
import warnings

from sphinx.util.inspect import safe_getattr

# we use the installed larray version ...
from larray import __version__
from larray import __file__ as _larray_path

# ... but we check it is the one from the same repository
_current_dir = os.path.dirname(__file__)
assert _larray_path == os.path.abspath(os.path.join(_current_dir, '..', '..', 'larray'))

_outdir = os.path.abspath(os.path.join(_current_dir, '..', 'build', 'check_api'))

_modules = ['larray', 'larray.random', 'larray.core.constants']
_exclude = ['absolute_import', 'division', 'print_function', 'renamed_to']


def _is_deprecated(obj):
    if not (inspect.isfunction(obj) or inspect.ismethod(obj)):
        return False
    return 'renamed_to' in obj.__qualname__


def _write_header(f, text, char='-', indent=''):
    f.write(indent + text + '\n')
    f.write(indent + char * len(text) + '\n')


class AbstractItem:
    def __init__(self, name):
        self.name = name

    def copy(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError()

    def insert_element(self, name, obj):
        raise NotImplementedError()

    def auto_discovery(self):
        raise NotImplementedError()

    def remove_deprecated(self):
        raise NotImplementedError()

    def only_deprecated(self):
        raise NotImplementedError()

    def diff(self, other):
        raise NotImplementedError()

    def write(self, ofile):
        raise NotImplementedError()


class ClassItem(AbstractItem):
    def __init__(self, class_):
        super().__init__(class_.__name__)
        self.class_ = class_
        self.attrs = {}
        self.methods = {}
        self.deprecated_methods = {}

    def copy(self):
        new_item = ClassItem(self.class_)
        new_item.attrs = self.attrs.copy()
        new_item.methods = self.methods.copy()
        new_item.deprecated_methods = self.deprecated_methods.copy()
        return new_item

    def __len__(self):
        return len(self.attrs) + len(self.methods) + len(self.deprecated_methods)

    def insert_element(self, name, obj):
        # See function build_py_coverage() in sphinx/sphinx/ext/coverage.py file
        # from sphinx github repository
        if name[0] == '_':
            # starts with an underscore, ignore it
            return
        if name in _exclude:
            return
        try:
            attr = safe_getattr(self.class_, name)
            if inspect.isfunction(attr):
                if _is_deprecated(obj):
                    self.deprecated_methods[name] = obj
                else:
                    self.methods[name] = obj
            else:
                self.attrs[name] = obj
        except AttributeError:
            pass

    def auto_discovery(self):
        for attr_name, attr_obj in inspect.getmembers(self.class_):
            self.insert_element(attr_name, attr_obj)

    def remove_deprecated(self):
        without_deprecated = self.copy()
        without_deprecated.deprecated_methods = {}
        return without_deprecated

    def only_deprecated(self):
        deprecated = ClassItem(self.class_)
        deprecated.deprecated_methods = self.deprecated_methods.copy()
        return deprecated

    def diff(self, other):
        if not isinstance(other, ClassItem):
            raise TypeError(f"Expect a {ClassItem.__name__} instance as argument. "
                            f"Got a {type(other).__name__} instance instead")
        diff_item = ClassItem(self.class_)
        diff_item.attrs = {k: v for k, v in self.attrs.items() if k not in other.attrs}
        diff_item.methods = {k: v for k, v in self.methods.items() if k not in other.methods}
        diff_item.deprecated_methods = {k: v for k, v in self.deprecated_methods.items()
                                        if k not in other.deprecated_methods}
        return diff_item

    def write(self, ofile):
        if len(self):
            indent = '   '
            _write_header(ofile, self.name, '=', indent=indent)
            if self.attrs:
                _write_header(ofile, 'Attributes', indent=indent)
                ofile.writelines(f'{indent} * {attr}\n' for attr in self.attrs.keys())
                ofile.write('\n')
            if self.methods:
                _write_header(ofile, 'Methods', indent=indent)
                ofile.writelines(f'{indent} * {method}\n' for method in self.methods.keys())
                ofile.write('\n')
            if self.deprecated_methods:
                _write_header(ofile, 'Deprecated Methods', indent=indent)
                ofile.writelines(f'{indent} * {method}\n' for method in self.deprecated_methods.keys())
                ofile.write('\n')
            ofile.write('\n')


class ModuleItem(AbstractItem):
    def __init__(self, module):
        super().__init__(module.__name__)
        self.module = module
        self.others = {}
        self.funcs = {}
        self.deprecated_items = {}
        self.classes = {}

    def copy(self):
        new_item = ModuleItem(self.module)
        new_item.others = self.others.copy()
        new_item.funcs = self.funcs.copy()
        new_item.deprecated_items = self.deprecated_items.copy()
        new_item.classes = self.classes.copy()
        return new_item

    def __len__(self):
        return len(self.others) + len(self.funcs) + len(self.deprecated_items) + len(self.classes)

    def insert_element(self, name, obj):
        # See function build_py_coverage() in sphinx/sphinx/ext/coverage.py file
        # from sphinx github repository

        # diverse module attributes are ignored:
        if name[0] == '_':
            # begins in an underscore
            return
        if not hasattr(obj, '__module__'):
            # cannot be attributed to a module
            return
        if name in _exclude:
            return

        if inspect.isfunction(obj):
            if _is_deprecated(obj):
                self.deprecated_items[name] = obj
            else:
                self.funcs[name] = obj
        elif inspect.isclass(obj):
            if name in self.classes:
                warnings.warn(f"Class '{name}' was already present in '{self.module.__name__}' module item "
                              f"and will be replaced")
            class_ = getattr(self.module, name)
            class_item = ClassItem(class_)
            self.classes[name] = class_item
        else:
            self.others[name] = obj

    def auto_discovery(self):
        for name, obj in inspect.getmembers(self.module):
            self.insert_element(name, obj)
            if inspect.isclass(obj):
                self.classes[name].auto_discovery()

    def remove_deprecated(self):
        without_deprecated = self.copy()
        without_deprecated.deprecated_items = {}
        without_deprecated.classes = {k: v.remove_deprecated() for k, v in self.classes.items()}
        return without_deprecated

    def only_deprecated(self):
        deprecated = ModuleItem(self.module)
        deprecated.deprecated_items = self.deprecated_items.copy()
        classes = {k: v.only_deprecated() for k, v in self.classes.items()}
        deprecated.classes = {k: v for k, v in classes.items() if len(v)}
        return deprecated

    def diff(self, other):
        if not isinstance(other, ModuleItem):
            raise TypeError(f"Expect a {ModuleItem.__name__} instance as argument. "
                            f"Got a {type(other).__name__} instance instead")
        diff_item = ModuleItem(self.module)
        diff_item.others = {k: v for k, v in self.others.items() if k not in other.others}
        diff_item.funcs = {k: v for k, v in self.funcs.items() if k not in other.funcs}
        diff_item.deprecated_items = {k: v for k, v in self.deprecated_items.items()
                                      if k not in other.deprecated_items}
        diff_item.classes = {k: v.diff(other.classes[k]) if k in other.classes else v
                             for k, v in self.classes.items()}
        return diff_item

    def write(self, ofile):
        if len(self):
            _write_header(ofile, 'module <' + self.name + '>', '~')
            ofile.write('\n')
            if self.others:
                _write_header(ofile, 'Miscellaneous', '=')
                ofile.writelines(f' * {other}\n' for other in self.others.keys())
                ofile.write('\n')
            if self.funcs:
                _write_header(ofile, 'Functions', '=')
                ofile.writelines(f' * {func}\n' for func in self.funcs.keys())
                ofile.write('\n')
            if self.deprecated_items:
                _write_header(ofile, 'Deprecated Functions or Classes', '=')
                ofile.writelines(f' * {func}\n' for func in self.deprecated_items.keys())
                ofile.write('\n')
            if self.classes:
                _write_header(ofile, 'Classes', '=')
                ofile.write('\n')
                for class_item in self.classes.values():
                    class_item.write(ofile)
                ofile.write('\n')
            ofile.write('\n')


def make_diff(left_api, right_api, include_deprecated=False):
    if not include_deprecated:
        left_api = {k: v.remove_deprecated() for k, v in left_api.items()}
        right_api = {k: v.remove_deprecated() for k, v in right_api.items()}

    diff_api = {}
    for left_module_name, left_module_item in left_api.items():
        if left_module_name in right_api:
            right_module_item = right_api[left_module_name]
            diff_api[left_module_name] = left_module_item.diff(right_module_item)
        else:
            diff_api[left_module_name] = left_module_item
    return diff_api


def get_module_item(module_name):
    try:
        module = importlib.import_module(module_name)
        return ModuleItem(module)
    except ImportError as err:
        print(f'module {module_name} could not be imported: {err}')
        return err


def get_public_api():
    public_api = {}
    for module_name in _modules:
        module_item = get_module_item(module_name)
        if isinstance(module_item, ModuleItem):
            module_item.auto_discovery()
        public_api[module_name] = module_item
    return public_api


# See file sphinx/sphinx/ext/autosummary/generate.py from sphinx github repository
def get_autosummary_api():
    import shutil
    from sphinx.ext.autosummary import import_by_name
    from sphinx.ext.autosummary.generate import DummyApplication, setup_documenters, generate_autosummary_docs

    sources = ['api.rst']
    output_dir = './tmp_generated'
    app = DummyApplication()
    setup_documenters(app)
    generate_autosummary_docs(sources, output_dir, app=app)

    autosummary_api = {}
    for module_name in _modules:
        module_item = get_module_item(module_name)
        if isinstance(module_item, ModuleItem):
            autosummary_api[module_name] = module_item

    for generated_rst_file in os.listdir(output_dir):
        qualname, ext = os.path.splitext(generated_rst_file)
        qualname, obj, parent, module_name = import_by_name(qualname)
        module_item = autosummary_api[module_name]
        name = qualname.split('.')[-1]
        if inspect.isclass(obj) and name in module_item.classes:
            continue
        if inspect.isclass(parent):
            class_name = parent.__name__
            if class_name not in module_item.classes:
                module_item.insert_element(class_name, parent)
            class_item = module_item.classes[class_name]
            class_item.insert_element(name, obj)
        else:
            module_item.insert_element(name, obj)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    return autosummary_api


def write_api(filepath, api, header='API', version=True):
    if not os.path.exists(_outdir):
        os.mkdir(_outdir)
    output_file = os.path.join(_outdir, filepath)

    failed = []
    with open(output_file, 'w') as ofile:
        if version:
            header = f'{header} [{__version__}]'
        _write_header(ofile, header, '~')
        ofile.write('\n')
        keys = sorted(api.keys())

        for module_name in keys:
            module_item = api[module_name]
            if not isinstance(module_item, ModuleItem):
                failed.append((module_name, module_item))
            else:
                module_item.write(ofile)

        if failed:
            _write_header(ofile, 'Modules that failed to import')
            ofile.writelines(f' * {module_name} -- {error}\n' for module_name, error in failed)


def get_items_from_api_doc():
    from sphinx.ext.autosummary import Autosummary
    from docutils.core import publish_doctree
    from docutils.parsers.rst import directives
    import docutils.nodes

    def add_item(item, api_doc_items):
        item = item.astext().strip().split()
        if len(item) > 0:
            # if item comes from a handwritten table (like the Exploring section of Axis in api.rst)
            # we select the text from the left column
            if isinstance(item, list):
                item = item[0]
            api_doc_items.append(item)

    api_doc_items = []
    directives.register_directive('autosummary', Autosummary)

    def cleanup_line(line):
        return line.replace(':attr:', '      ').replace('`', ' ')

    def check_line(line):
        return len(line) and not (line.startswith('..') or ':' in line)

    with open('./api.rst', mode='r') as f:
        content = [cleanup_line(line) for line in f.readlines()]
        content = '\n'.join([line for line in content if check_line(line)])
        document = publish_doctree(content)
        nodes = list(document)
        for node in nodes:
            if isinstance(node, docutils.nodes.block_quote):
                for item in node.children:
                    add_item(item, api_doc_items)
            if isinstance(node, docutils.nodes.table):
                for item in node[0][2].children:
                    add_item(item, api_doc_items)
    return api_doc_items


if __name__ == '__main__':
    public_api = get_public_api()
    write_api('public_api.txt', public_api, header='PUBLIC API')

    public_api_only_deprecated = {k: v.only_deprecated() for k, v in public_api.items()}
    write_api('public_api_only_deprecated.txt', public_api_only_deprecated, header='DEPRECATED API')

    api_reference = get_autosummary_api()
    write_api('api_reference.txt', api_reference, header='API REFERENCE')

    api_reference_only_deprecated = {k: v.only_deprecated() for k, v in api_reference.items()}
    write_api('api_reference_only_deprecated.txt', api_reference_only_deprecated,
              header='DEPRECATED ITEMS IN API REFERENCE')

    missing_in_api_ref = make_diff(public_api, api_reference)
    write_api('missing_api_items_in_doc.txt', missing_in_api_ref, header='MISSING ITEMS IN API REFERENCE')
