import warnings
from pathlib import Path
from typing import Union

try:
    import xlwings as xw
except ImportError:
    xw = None

from larray.util.misc import _positive_integer
from larray.core.group import _translate_sheet_name
from larray.core.array import asarray, zip_array_items
from larray.example import load_example_data, EXAMPLE_EXCEL_TEMPLATES_DIR  # noqa: F401 (only used in doctests)


_default_items_size = {}


def _validate_template_filename(filename: Union[str, Path]) -> Path:
    if isinstance(filename, str):
        filename = Path(filename)
    suffix = filename.suffix
    if not suffix:
        suffix = '.crtx'
    if suffix != '.crtx':
        raise ValueError(f"Extension for the excel template file must be '.crtx' instead of {suffix}")
    return filename.with_suffix(suffix)


class AbstractReportItem:
    def __init__(self, template_dir=None, template=None, graphs_per_row=1):
        self.template_dir = template_dir
        self.template = template
        self.default_items_size = _default_items_size.copy()
        self.graphs_per_row = graphs_per_row

    @property
    def template_dir(self):
        r"""
        Set the path to the directory containing the Excel template files (with '.crtx' extension).

        This method is mainly useful if your template files are located in several directories,
        otherwise pass the template directory directly the ExcelReport constructor.

        Parameters
        ----------
        template_dir : str or Path
            Path to the directory containing the Excel template files.

        See Also
        --------
        set_graph_template

        Examples
        --------
        >>> report = ExcelReport(EXAMPLE_EXCEL_TEMPLATES_DIR)
        >>> # ... add some graphs using template files from 'C:\excel_templates_dir'
        >>> report.template_dir = r'C:\other_templates_dir' # doctest: +SKIP
        >>> # ... add some graphs using template files from 'C:\other_templates_dir'
        """
        return self._template_dir

    @template_dir.setter
    def template_dir(self, template_dir):
        if template_dir is not None:
            if isinstance(template_dir, str):
                template_dir = Path(template_dir)
            if not isinstance(template_dir, Path):
                raise TypeError(f"Expected a string or a pathlib.Path object. "
                                f"Got an object of type {type(template_dir).__name__} instead.")
            if not template_dir.is_dir():
                raise ValueError(f"The directory {template_dir} could not be found.")
        self._template_dir = template_dir

    @property
    def template(self):
        r"""
        Set a default Excel template file.

        Parameters
        ----------
        template : str or Path
            Name of the template to be used as default template.
            The extension '.crtx' will be added if not given.
            The full path to the template file must be given if no template directory has been set.

        Examples
        --------
        >>> demo = load_example_data('demography_eurostat')

        Passing the name of the template (only if a template directory has been set)

        >>> report = ExcelReport(EXAMPLE_EXCEL_TEMPLATES_DIR)
        >>> report.template = 'Line'

        >>> sheet_population = report.new_sheet('Population')
        >>> sheet_population.add_graph(demo.population['Belgium'],'Belgium')

        Passing the full path of the template file

        >>> # if no default template directory has been set
        >>> # or if the new template is located in another directory,
        >>> # you must provide the full path
        >>> sheet_population.template = r'C:\other_templates_dir\Line_Marker.crtx' # doctest: +SKIP
        >>> sheet_population.add_graph(demo.population['Germany'],'Germany') # doctest: +SKIP
        """
        return self._template

    @template.setter
    def template(self, template):
        if template is not None:
            if self.template_dir is None:
                raise RuntimeError("Please set 'template_dir' first")
            filename = _validate_template_filename(template)
            template = self.template_dir / filename
        self._template = template

    def set_item_default_size(self, kind, width=None, height=None):
        r"""
        Override the default 'width' and 'height' values for the given kind of item.

        A new value must be provided at least for 'width' or 'height'.

        Parameters
        ----------
        kind : str
            kind of item for which default values of 'width' and/or 'height' are modified.
            Currently available kinds are 'title' and 'graph'.
        width : int, optional
            new default width value.
        height : int, optional
            new default height value.

        Examples
        --------
        >>> report = ExcelReport()
        >>> report.set_item_default_size('graph', width=450, height=250)
        """
        if width is None and height is None:
            raise ValueError("No value provided for both 'width' and 'heigth'. "
                             "Please provide one for at least 'width' or 'heigth'")
        if kind not in self.default_items_size:
            item_types = sorted(self.default_items_size.keys())
            raise ValueError(f"Item type {kind} is not registered. Please choose in list {item_types}")
        if width is None:
            width = self.default_items_size[kind].width
        if height is None:
            height = self.default_items_size[kind].height
        self.default_items_size[kind] = ItemSize(width, height)

    @property
    def graphs_per_row(self):
        r"""
        Default number of graphs per row.

        Parameters
        ----------
        graphs_per_row: int

        See Also
        --------
        ReportSheet.newline
        """
        return self._graphs_per_row

    @graphs_per_row.setter
    def graphs_per_row(self, graphs_per_row):
        _positive_integer(graphs_per_row)
        self._graphs_per_row = graphs_per_row


class AbstractReportSheet(AbstractReportItem):
    r"""
    Represents a sheet dedicated to contains only graphical items (title banners, graphs).

    See :py:obj:`ExcelReport` for use cases.

    Parameters
    ----------
    template_dir : str or Path, optional
        Path to the directory containing the Excel template files (with a '.crtx' extension).
        Defaults to None.
    template : str or Path, optional
        Name of the template to be used as default template.
        The extension '.crtx' will be added if not given.
        The full path to the template file must be given if no template directory has been set.
        Defaults to None.
    graphs_per_row : int, optional
        Default number of graphs per row. Defaults to 1.

    See Also
    --------
    ExcelReport
    """

    def add_title(self, title, width=None, height=None, fontsize=11):
        r"""
        Add a title item to the current sheet.

        Note that the current method only add a new item to the list of items to be generated.
        The report Excel file is generated only when the :py:obj:`~ExcelReport.to_excel` is called.

        Parameters
        ----------
        title : str
            Text to write in the title item.
        width : int, optional
            width of the title item. The current default value is used if None
            (see :py:obj:`~ExcelReport.set_item_default_size`). Defaults to None.
        height : int, optional
            height of the title item. The current default value is used if None
            (see :py:obj:`~ExcelReport.set_item_default_size`). Defaults to None.
        fontsize : int, optional
            fontsize of the displayed text. Defaults to 11.

        Examples
        --------
        >>> report = ExcelReport()

        >>> first_sheet = report.new_sheet('First_sheet')
        >>> first_sheet.add_title('Title banner with default width, height and fontsize')
        >>> first_sheet.add_title('Larger title banner', width=1200, height=100)
        >>> first_sheet.add_title('Bigger fontsize', fontsize=13)

        >>> # do not forget to call 'to_excel' to create the report file
        >>> report.to_excel('Report.xlsx')
        """
        pass

    def add_graph(self, data, title=None, template=None, width=None, height=None, min_y=None, max_y=None,
                  xticks_spacing=None, customize_func=None, customize_kwargs=None):
        r"""
        Add a graph item to the current sheet.

        Note that the current method only add a new item to the list of items to be generated.
        The report Excel file is generated only when the :py:obj:`~ExcelReport.to_excel` is called.

        Parameters
        ----------
        data : 1D or 2D array-like
            1D or 2D array representing the data associated with the graph.
            The first row represents the abscissa labels.
            Each additional row represents a new series and must start with the name of the current series.
        title : str, optional
            title of the graph. Defaults to None.
        template : str or Path, optional
            name of the template to be used to generate the graph.
            The full path to the template file must be provided if no template directory has not been set
            or if the template file belongs to another directory.
            Defaults to the defined template (see :py:obj:`~ExcelReport.set_graph_template`).
        width : int, optional
            width of the title item. The current default value is used if None
            (see :py:obj:`~ExcelReport.set_item_default_size`). Defaults to None.
        height : int, optional
            height of the title item. The current default value is used if None
            (see :py:obj:`~ExcelReport.set_item_default_size`). Defaults to None.
        min_y: int, optional
            minimum value for the Y axis.
        max_y: int, optional
            maximum value for the Y axis.
        xticks_spacing: int, optional
            space interval between two ticks along the X axis.
        customize_func: function, optional
            user defined function to personalize the graph.
            The function must take the Chart object as first argument.
            All keyword arguments defined in customize_kwargs are passed to the function at call.
        customize_kwargs: dict, optional
            keywords arguments passed to the function `customize_func` at call.

        Examples
        --------
        >>> demo = load_example_data('demography_eurostat')
        >>> report = ExcelReport(EXAMPLE_EXCEL_TEMPLATES_DIR)

        >>> sheet_be = report.new_sheet('Belgium')

        Specifying the 'template'

        >>> sheet_be.add_graph(demo.population['Belgium'], 'Population', template='Line')

        Specifying the 'template', 'width' and 'height' values

        >>> sheet_be.add_graph(demo.births['Belgium'], 'Births', template='Line', width=450, height=250)

        Setting a default template

        >>> sheet_be.template = 'Line_Marker'
        >>> sheet_be.add_graph(demo.deaths['Belgium'], 'Deaths')

        Specify the mininum and maximum values for the Y axis

        >>> sheet_be.add_graph(demo.population['Belgium'],
        ...                    'Population (min/max Y axis = 5/6 millions)',
        ...                     min_y=5e6, max_y=6e6)

        Specify the interval between two ticks (X axis)

        >>> sheet_be.add_graph(demo.population['Belgium'], 'Population (every 2 years)', xticks_spacing=2)

        Dumping the report Excel file

        >>> # do not forget to call 'to_excel' to create the report file
        >>> report.to_excel('Demography_Report.xlsx')
        """
        pass

    def add_graphs(self, array_per_title, axis_per_loop_variable, template=None, width=None, height=None,
                   graphs_per_row=1, min_y=None, max_y=None, xticks_spacing=None, customize_func=None,
                   customize_kwargs=None):
        r"""
        Add multiple graph items to the current sheet.

        This method is mainly useful when multiple graphs are generated by iterating over one or several axes of an
        array (see examples below).
        The report Excel file is generated only when the :py:obj:`~ExcelReport.to_excel` is called.

        Parameters
        ----------
        array_per_title: dict
            dictionary containing pairs (title template, array).
        axis_per_loop_variable: dict
            dictionary containing pairs (variable used in the title template, axis).
        template : str or Path, optional
            name of the template to be used to generate the graph.
            The full path to the template file must be provided if no template directory has not been set
            or if the template file belongs to another directory.
            Defaults to the defined template (see :py:obj:`~ExcelReport.set_graph_template`).
        width : int, optional
            width of the title item. The current default value is used if None
            (see :py:obj:`~ExcelReport.set_item_default_size`). Defaults to None.
        height : int, optional
            height of the title item. The current default value is used if None
            (see :py:obj:`~ExcelReport.set_item_default_size`). Defaults to None.
        graphs_per_row: int, optional
            Number of graphs per row. Defaults to 1.
        min_y: int, optional
            minimum value for the Y axis.
        max_y: int, optional
            maximum value for the Y axis.
        xticks_spacing: int, optional
            space interval between two ticks along the X axis.
        customize_func: function, optional
            user defined function to personalize the graph.
            The function must take the Chart object as first argument.
            All keyword arguments defined in customize_kwargs are passed to the function at call.
        customize_kwargs: dict, optional
            keywords arguments passed to the function `customize_func` at call.

        Examples
        --------
        >>> demo = load_example_data('demography_eurostat')
        >>> report = ExcelReport(EXAMPLE_EXCEL_TEMPLATES_DIR)

        >>> sheet_population = report.new_sheet('Population')
        >>> population = demo.population

        Generate a new graph for each combination of gender and year

        >>> sheet_population.add_graphs(
        ...     {'Population of {gender} by country in {year}': population},
        ...     {'gender': population.gender, 'year': population.time},
        ...     template='line', width=450, height=250, graphs_per_row=2)

        Specify the mininum and maximum values for the Y axis

        >>> sheet_population.add_graphs({'Population of {gender} by country for the year {year}': population},
        ...                      {'gender': population.gender, 'year': population.time},
        ...                      template='line', width=450, height=250, graphs_per_row=2, min_y=0, max_y=50e6)

        Specify the interval between two ticks (X axis)

        >>> sheet_population.add_graphs({'Population of {gender} by country for the year {year}': population},
        ...                      {'gender': population.gender, 'year': population.time},
        ...                      template='line', width=450, height=250, graphs_per_row=2, xticks_spacing=2)

        >>> # do not forget to call 'to_excel' to create the report file
        >>> report.to_excel('Demography_Report.xlsx')
        """
        pass

    def newline(self):
        r"""
        Force a new row of graphs.
        """
        pass


class AbstractExcelReport(AbstractReportItem):
    r"""
    Automate the generation of multiple graphs in an Excel file.

    The ExcelReport instance is initially populated with information
    (data, title, destination sheet, template, size) required to create the graphs.
    Once all information has been provided, the :py:obj:`~ExcelReport.to_excel` method
    is called to generate an Excel file with all graphs in one step.

    Parameters
    ----------
    template_dir : str or Path, optional
        Path to the directory containing the Excel template files (with a '.crtx' extension).
        Defaults to None.
    template : str or Path, optional
        Name of the template to be used as default template.
        The extension '.crtx' will be added if not given.
        The full path to the template file must be given if no template directory has been set.
        Defaults to None.
    graphs_per_row: int, optional
        Default number of graphs per row.
        Defaults to 1.

    Notes
    -----
    The data associated with all graphical items is dumped in the same sheet named '__data__'.

    Examples
    --------
    >>> demo = load_example_data('demography_eurostat')
    >>> report = ExcelReport(EXAMPLE_EXCEL_TEMPLATES_DIR)

    Set a new destination sheet

    >>> sheet_be = report.new_sheet('Belgium')

    Add a new title item

    >>> sheet_be.add_title('Population, births and deaths')

    Add a new graph item (each new graph is placed right to previous one unless you use newline() or add_title())

    >>> # using default 'width' and 'height' values
    >>> sheet_be.add_graph(demo.population['Belgium'], 'Population', template='Line')
    >>> # specifying the 'width' and 'height' values
    >>> sheet_be.add_graph(demo.births['Belgium'], 'Births', template='Line', width=450, height=250)

    Override the default 'width' and 'height' values for graphs

    >>> sheet_be.set_item_default_size('graph', width=450, height=250)
    >>> # add a new graph with the new default 'width' and 'height' values
    >>> sheet_be.add_graph(demo.deaths['Belgium'], 'Deaths')

    Set a default template for all next graphs

    >>> # if a default template directory has been set, just pass the name
    >>> sheet_be.template = 'Line'
    >>> # otherwise, give the full path to the template file
    >>> sheet_be.template = r'C:\other_template_dir\Line_Marker.crtx' # doctest: +SKIP
    >>> # add a new graph with the default template
    >>> sheet_be.add_graph(demo.population['Belgium', 'Female'], 'Population - Female')
    >>> sheet_be.add_graph(demo.population['Belgium', 'Male'], 'Population - Male')

    Specify the number of graphs per row

    >>> sheet_countries = report.new_sheet('All countries')

    >>> sheet_countries.graphs_per_row = 2
    >>> for combined_labels, subset in demo.population.items(('time', 'gender')):
    ...    title = ' - '.join([str(label) for label in combined_labels])
    ...    sheet_countries.add_graph(subset, title)

    Force a new row of graphs

    >>> sheet_countries.newline()

    Add multiple graphs at once (add a new graph for each combination of gender and year)

    >>> sheet_countries.add_graphs({'Population of {gender} by country in {year}': population},
    ...                            {'gender': population.gender, 'year': population.time},
    ...                            template='line', width=450, height=250, graphs_per_row=2)

    Generate the report Excel file

    >>> report.to_excel('Demography_Report.xlsx')
    """

    def new_sheet(self, sheet_name):
        r"""
        Add a new empty output sheet.

        This sheet will contain only graphical elements, all data are exported
        to a dedicated separate sheet.

        Parameters
        ----------
        sheet_name : str
            name of the current sheet.

        Returns
        -------
        sheet: ReportSheet

        Examples
        --------
        >>> demo = load_example_data('demography_eurostat')
        >>> report = ExcelReport(EXAMPLE_EXCEL_TEMPLATES_DIR)

        >>> # prepare new output sheet named 'Belgium'
        >>> sheet_be = report.new_sheet('Belgium')

        >>> # add graph to the output sheet 'Belgium'
        >>> sheet_be.add_graph(demo.population['Belgium'], 'Population', template='Line')
        """
        pass

    def sheet_names(self):
        r"""
        Return the names of the output sheets.

        Examples
        --------
        >>> report = ExcelReport()
        >>> sheet_population = report.new_sheet('Pop')
        >>> sheet_births = report.new_sheet('Births')
        >>> sheet_deaths = report.new_sheet('Deaths')
        >>> report.sheet_names()
        ['Pop', 'Births', 'Deaths']
        """
        pass

    def to_excel(self, filepath, data_sheet_name='__data__', overwrite=True):
        r"""
        Generate the report Excel file.

        Parameters
        ----------
        filepath : str or Path
            Path of the report file for the dump.
        data_sheet_name : str, optional
            name of the Excel sheet where all data associated with items is dumped.
            Defaults to '__data__'.
        overwrite : bool, optional
            whether to overwrite an existing report file.
            Defaults to True.

        Examples
        --------
        >>> demo = load_example_data('demography_eurostat')
        >>> report = ExcelReport(EXAMPLE_EXCEL_TEMPLATES_DIR)
        >>> report.template = 'Line_Marker'

        >>> for c in demo.country:
        ...     sheet_country = report.new_sheet(c)
        ...     sheet_country.add_graph(demo.population[c], 'Population')
        ...     sheet_country.add_graph(demo.births[c], 'Births')
        ...     sheet_country.add_graph(demo.deaths[c], 'Deaths')

        Basic usage

        >>> report.to_excel('Demography_Report.xlsx')

        Alternative data sheet name

        >>> report.to_excel('Demography_Report.xlsx', data_sheet_name='Data Tables') # doctest: +SKIP

        Check if ouput file already exists

        >>> report.to_excel('Demography_Report.xlsx', overwrite=False) # doctest: +SKIP
        Traceback (most recent call last):
        ...
        ValueError: Sheet named 'Belgium' already present in workbook
        """
        pass


if xw is not None:
    from xlwings.constants import LegendPosition, HAlign, VAlign, ChartType, RowCol, AxisType, Constants
    from larray.inout.xw_excel import open_excel

    class ItemSize:
        def __init__(self, width, height):
            self.width = width
            self.height = height

        @property
        def width(self):
            return self._width

        @width.setter
        def width(self, width):
            _positive_integer(width)
            self._width = width

        @property
        def height(self):
            return self._height

        @height.setter
        def height(self, height):
            _positive_integer(height)
            self._height = height


    class ExcelTitleItem(ItemSize):

        _default_size = ItemSize(1000, 50)

        def __init__(self, text, fontsize, top, left, width, height):
            ItemSize.__init__(self, width, height)
            self.top = top
            self.left = left
            self.text = str(text)
            _positive_integer(fontsize)
            self.fontsize = fontsize

        def dump(self, sheet, data_sheet, row):
            data_cells = data_sheet.Cells

            # add title in data sheet
            data_cells(row, 1).Value = self.text

            # generate title banner in destination sheet
            msoShapeRectangle = 1
            msoThemeColorBackground1 = 14
            sheet_shapes = sheet.Shapes
            shp = sheet_shapes.AddShape(Type=msoShapeRectangle, Left=self.left, Top=self.top,
                                        Width=self.width, Height=self.height)
            fill = shp.Fill
            fill.ForeColor.ObjectThemeColor = msoThemeColorBackground1
            fill.Solid()
            shp.Line.Visible = False
            frame = shp.TextFrame
            chars = frame.Characters()
            chars.Text = self.text
            font = chars.Font
            font.Color = 1
            font.Bold = True
            font.Size = self.fontsize
            frame.HorizontalAlignment = HAlign.xlHAlignLeft
            frame.VerticalAlignment = VAlign.xlVAlignCenter
            shp.SetShapesDefaultProperties()
            # update and return current row position in data sheet (+1 for title +1 for blank line)
            return row + 2

    _default_items_size['title'] = ExcelTitleItem._default_size

    class ExcelGraphItem(ItemSize):

        _default_size = ItemSize(427, 230)

        def __init__(self, data, title, template, top, left, width, height, min_y, max_y,
                     xticks_spacing, customize_func, customize_kwargs):
            ItemSize.__init__(self, width, height)
            self.top = top
            self.left = left
            self.title = str(title) if title is not None else None
            data = asarray(data)
            if not (1 <= data.ndim <= 2):
                raise ValueError(f"Expected 1D or 2D array for data argument. Got array of dimensions {data.ndim}")
            self.data = data
            if template is not None:
                template = Path(template)
                if not template.is_file():
                    raise ValueError(f"Could not find template file {template}")
            self.template = template
            self.min_y = min_y
            self.max_y = max_y
            self.xticks_spacing = xticks_spacing
            if customize_func is not None and not callable(customize_func):
                raise TypeError(f"Expected a function for the argument 'customize_func'. "
                                f"Got object of type {type(customize_func).__name__} instead.")
            self.customize_func = customize_func
            self.customize_kwargs = customize_kwargs

        def dump(self, sheet, data_sheet, row):
            data_range = data_sheet.Range
            data_cells = data_sheet.Cells

            # write graph title in data sheet
            data_cells(row, 1).Value = self.title
            row += 1

            # dump data to make the graph in data sheet
            data = self.data
            nb_series = 1 if data.ndim == 1 else data.shape[0]
            nb_xticks = data.size if data.ndim == 1 else data.shape[1]
            last_row, last_col = row + nb_series, nb_xticks + 1
            data_range(data_cells(row, 1), data_cells(last_row, last_col)).Value = data.dump(na_repr=None)
            data_cells(row, 1).Value = ''

            # generate graph in destination sheet
            sheet_charts = sheet.ChartObjects()
            obj = sheet_charts.Add(self.left, self.top, self.width, self.height)
            obj_chart = obj.Chart
            source = data_range(data_cells(row, 1), data_cells(last_row, last_col))
            obj_chart.SetSourceData(source)
            obj_chart.ChartType = ChartType.xlLine
            # title
            if self.title is not None:
                obj_chart.HasTitle = True
                obj_chart.ChartTitle.Caption = self.title
            # legend
            obj_chart.Legend.Position = LegendPosition.xlLegendPositionBottom
            # template
            if self.template is not None:
                obj_chart.ApplyChartTemplate(self.template)
            # min - max on Y axis
            if self.min_y is not None:
                obj_chart.Axes(AxisType.xlValue).MinimumScale = self.min_y
            if self.max_y is not None:
                obj_chart.Axes(AxisType.xlValue).MaximumScale = self.max_y
            # xticks_spacing
            if self.xticks_spacing is not None:
                obj_chart.Axes(AxisType.xlCategory).TickLabelSpacing = self.xticks_spacing
                obj_chart.Axes(AxisType.xlCategory).TickMarkSpacing = self.xticks_spacing
                obj_chart.Axes(AxisType.xlCategory).TickLabelPosition = Constants.xlLow
            # user's function (to apply on remaining kwargs)
            if self.customize_func is not None:
                self.customize_func(obj_chart, **self.customize_kwargs)
            # flagflip
            if nb_series > 1 and nb_xticks == 1:
                obj_chart.PlotBy = RowCol.xlRows
            # update and return current row position
            return row + nb_series + 2

    _default_items_size['graph'] = ExcelGraphItem._default_size

    class ReportSheet(AbstractReportSheet):
        def __init__(self, excel_report, name, template_dir=None, template=None, graphs_per_row=1):
            name = _translate_sheet_name(name)
            self.excel_report = excel_report
            self.name = name
            self.items = []
            self.top = 0
            self.left = 0
            self.position_in_row = 1
            self.curline_height = 0
            if template_dir is None:
                template_dir = excel_report.template_dir
            if template is None:
                template = excel_report.template
            AbstractReportSheet.__init__(self, template_dir, template, graphs_per_row)

        def add_title(self, title, width=None, height=None, fontsize=11):
            if width is None:
                width = self.default_items_size['title'].width
            if height is None:
                height = self.default_items_size['title'].height
            self.newline()
            self.items.append(ExcelTitleItem(title, fontsize, self.top, 0, width, height))
            self.top += height

        def add_graph(self, data, title=None, template=None, width=None, height=None, min_y=None, max_y=None,
                      xticks_spacing=None, customize_func=None, customize_kwargs=None):
            if width is None:
                width = self.default_items_size['graph'].width
            if height is None:
                height = self.default_items_size['graph'].height
            if template is not None:
                self.template = template
            template = self.template
            if self.graphs_per_row is not None and self.position_in_row > self.graphs_per_row:
                self.newline()
            self.items.append(ExcelGraphItem(data, title, template, self.top, self.left, width, height,
                                             min_y, max_y, xticks_spacing, customize_func, customize_kwargs))
            self.left += width
            self.curline_height = max(self.curline_height, height)
            self.position_in_row += 1

        def add_graphs(self, array_per_title, axis_per_loop_variable, template=None, width=None, height=None,
                       graphs_per_row=1, min_y=None, max_y=None, xticks_spacing=None, customize_func=None,
                       customize_kwargs=None):
            loop_variable_names = axis_per_loop_variable.keys()
            axes = tuple(axis_per_loop_variable.values())
            titles = array_per_title.keys()
            arrays = array_per_title.values()
            if graphs_per_row is not None:
                previous_graphs_per_row = self.graphs_per_row
                self.graphs_per_row = graphs_per_row
            if self.position_in_row > 1:
                self.newline()
            for loop_variable_values, arrays_chunk in zip_array_items(arrays, axes=axes):
                loop_variables_dict = dict(zip(loop_variable_names, loop_variable_values))
                for title_template, array_chunk in zip(titles, arrays_chunk):
                    title = title_template.format(**loop_variables_dict)
                    self.add_graph(array_chunk, title, template, width, height, min_y, max_y, xticks_spacing,
                                   customize_func, customize_kwargs)
            if graphs_per_row is not None:
                self.graphs_per_row = previous_graphs_per_row

        def newline(self):
            self.top += self.curline_height
            self.curline_height = 0
            self.left = 0
            self.position_in_row = 1

        def _to_excel(self, workbook, data_row):
            # use first sheet as data sheet
            data_sheet = workbook.Worksheets(1)
            data_cells = data_sheet.Cells
            # write destination sheet name in data sheet
            data_cells(data_row, 1).Value = self.name
            data_row += 2

            # create new empty sheet in workbook (will contain output graphical items)
            # Hack, since just specifying "After" is broken in certain environments
            # see: https://stackoverflow.com/questions/40179804/adding-excel-sheets-to-end-of-workbook
            dest_sheet = workbook.Worksheets.Add(Before=None, After=workbook.Sheets(workbook.Sheets.Count))
            dest_sheet.Name = self.name
            # for each item, dump data + generate associated graphical items
            for item in self.items:
                data_row = item.dump(dest_sheet, data_sheet, data_row)
            # reset
            self.top = 0
            self.left = 0
            self.curline_height = 0
            # return current row in data sheet
            return data_row

    # TODO : add a new section about this class in the tutorial
    class ExcelReport(AbstractExcelReport):
        def __init__(self, template_dir=None, template=None, graphs_per_row=1):
            AbstractExcelReport.__init__(self, template_dir, template, graphs_per_row)
            self.sheets = {}

        def sheet_names(self):
            return [sheet_name for sheet_name in self.sheets.keys()]

        def __getitem__(self, key):
            return self.sheets[key]

        # TODO : Do not implement __setitem__ and move code below to new_sheet()?
        def __setitem__(self, key, value, warn_stacklevel=2):
            if not isinstance(value, ReportSheet):
                raise ValueError(f"Expected ReportSheet object. Got {type(value).__name__} object instead.")
            if key in self.sheet_names():
                warnings.warn(f"Sheet '{key}' already exists in the report and will be reset",
                              stacklevel=warn_stacklevel)
            self.sheets[key] = value

        def __delitem__(self, key):
            del self.sheets[key]

        def __repr__(self):
            return f'sheets: {self.sheet_names()}'

        def new_sheet(self, sheet_name):
            sheet = ReportSheet(self, sheet_name, self.template_dir, self.template, self.graphs_per_row)
            self.__setitem__(sheet_name, sheet, warn_stacklevel=3)
            return sheet

        def to_excel(self, filepath, data_sheet_name='__data__', overwrite=True):
            with open_excel(filepath, overwrite_file=overwrite) as wb:
                # from here on, we use pure win32com objects instead of
                # larray.excel or xlwings objects as this is faster
                xl_wb = wb.api

                # rename first sheet
                xl_wb.Worksheets(1).Name = data_sheet_name

                # dump items for each output sheet
                data_sheet_row = 1
                for sheet in self.sheets.values():
                    data_sheet_row = sheet._to_excel(xl_wb, data_sheet_row)
                wb.save()
                # reset
                self.sheets.clear()
else:
    class ReportSheet(AbstractReportSheet):
        def __init__(self):
            raise Exception("ReportSheet class cannot be instantiated because xlwings is not installed")

    class ExcelReport(AbstractExcelReport):
        def __init__(self):
            raise Exception("ExcelReport class cannot be instantiated because xlwings is not installed")


ExcelReport.__doc__ = AbstractExcelReport.__doc__
ReportSheet.__doc__ = AbstractReportSheet.__doc__
