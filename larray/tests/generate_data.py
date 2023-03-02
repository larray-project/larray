import os

from larray import ndtest, open_excel, Session, X


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def generate_tests_files():
    tests = {'1d': 3,
             '2d': "a=1..3; b=b0,b1",
             '2d_classic': "a=a0..a2;b=b0..b2",
             '3d': "a=1..3; b=b0,b1; c=c0..c2",
             'int_labels': "a=0..2; b=0..2; c=0..2",
             'missing_values': "a=1..3; b=b0,b1; c=c0..c2",
             'unsorted': "a=3..1; b=b1,b0; c=c2..c0",
             'position': "a=1..3; b=b0,b1; c=c0..c2"}

    wb = open_excel(os.path.join(DATA_DIR, 'test.xlsx'), overwrite_file=True)
    wb_narrow = open_excel(os.path.join(DATA_DIR, 'test_narrow.xlsx'), overwrite_file=True)

    for name, dim in tests.items():
        arr = ndtest(dim)
        if name == '2d_classic':
            df = arr.to_frame(fold_last_axis_name=False)
            # wide format
            df.to_csv(os.path.join(DATA_DIR, f'test{name}.csv'), sep=',', na_rep='')
            wb[name] = ''
            wb[name]['A1'].options().value = df
            # narrow format
            df = arr.to_series(name='value')
            df.to_csv(os.path.join(DATA_DIR, f'test{name}_narrow.csv'), sep=',', na_rep='', header=True)
            wb_narrow[name] = ''
            wb_narrow[name]['A1'].options().value = df
        elif name == 'missing_values':
            df = arr.to_frame(fold_last_axis_name=True)
            # wide format
            df = df.drop([(2, 'b0'), (3, 'b1')])
            df.to_csv(os.path.join(DATA_DIR, f'test{name}.csv'), sep=',', na_rep='')
            wb[name] = ''
            wb[name]['A1'].options().value = df
            # narrow format
            df = arr.to_series(name='value')
            df = df.drop([(2, 'b0'), (2, 'b1', 'c1'), (3, 'b1')])
            df.to_csv(os.path.join(DATA_DIR, f'test{name}_narrow.csv'), sep=',', na_rep='', header=True)
            wb_narrow[name] = ''
            wb_narrow[name]['A1'].options().value = df
        elif name == 'position':
            # wide format
            wb[name] = ''
            wb[name]['D3'] = arr.dump()
            # narrow format
            wb_narrow[name] = ''
            wb_narrow[name]['D3'] = arr.dump(wide=False)
        else:
            # wide format
            arr.to_csv(os.path.join(DATA_DIR, f'test{name}.csv'))
            wb[name] = arr.dump()
            # narrow format
            arr.to_csv(os.path.join(DATA_DIR, f'test{name}_narrow.csv'), wide=False)
            wb_narrow[name] = arr.dump(wide=False)

    wb.save()
    wb.close()
    wb_narrow.save()
    wb_narrow.close()


def generate_example_files(csv=True, excel=True, hdf5=True):
    from larray_eurostat import eurostat_get

    def prepare_eurostat_data(dataset_name, countries):
        arr = eurostat_get(dataset_name)[X.unit['NR'], X.age['TOTAL'], X.sex['M,F']]
        arr = arr[X.time[::-1]][2013:2017]
        arr = arr.rename('sex', 'gender')
        arr = arr.set_labels(gender='Male,Female')
        arr = arr.rename('geo', 'country')
        country_codes = list(countries.keys())
        country_names = list(countries.values())
        if dataset_name == 'migr_imm1ctz':
            # example of an array with ambiguous axes
            arr = arr['COMPLET', X.citizen[country_codes], X.country[country_codes]].astype(int)
            arr = arr.rename('citizen', 'citizenship')
            arr = arr.set_labels('citizenship', country_names)
            arr = arr.set_labels('country', country_names)
            arr = arr.transpose('country', 'citizenship', 'gender', 'time')
        else:
            arr = arr[country_codes].astype(int)
            arr = arr.set_labels('country', country_names)
            arr = arr.transpose('country', 'gender', 'time')
        return arr

    countries = {'BE': 'Belgium', 'FR': 'France', 'DE': 'Germany'}
    benelux = {'BE': 'Belgium', 'LU': 'Luxembourg', 'NL': 'Netherlands'}

    # Arrays
    population = prepare_eurostat_data('demo_pjan', countries)
    population.meta.title = 'Population on 1 January by age and sex'
    population.meta.source = 'table demo_pjan from Eurostat'
    # ----
    population_benelux = prepare_eurostat_data('demo_pjan', benelux)
    population_benelux.meta.title = 'Population on 1 January by age and sex (Benelux)'
    population_benelux.meta.source = 'table demo_pjan from Eurostat'
    # ----
    population_5_countries = population.append('country', population_benelux[['Luxembourg', 'Netherlands']])
    population_5_countries.meta.title = 'Population on 1 January by age and sex (Benelux + France + Germany)'
    population_5_countries.meta.source = 'table demo_pjan from Eurostat'
    # ----
    births = prepare_eurostat_data('demo_fasec', countries)
    births.meta.title = "Live births by mother's age and newborn's sex"
    births.meta.source = 'table demo_fasec from Eurostat'
    # ----
    deaths = prepare_eurostat_data('demo_magec', countries)
    deaths.meta.title = 'Deaths by age and sex'
    deaths.meta.source = 'table demo_magec from Eurostat'
    # ----
    immigration = prepare_eurostat_data('migr_imm1ctz', benelux)
    immigration.meta.title = 'Immigration by age group, sex and citizenship'
    immigration.meta.source = 'table migr_imm1ctz from Eurostat'

    # Groups
    even_years = population.time[2014::2] >> 'even_years'
    odd_years = population.time[2013::2] >> 'odd_years'

    # Session
    ses = Session({'country': population.country, 'country_benelux': immigration.country,
                   'citizenship': immigration.citizenship,
                   'gender': population.gender, 'time': population.time,
                   'even_years': even_years, 'odd_years': odd_years,
                   'population': population, 'population_benelux': population_benelux,
                   'population_5_countries': population_5_countries,
                   'births': births, 'deaths': deaths, 'immigration': immigration})
    ses.meta.title = 'Demographic datasets for a small selection of countries in Europe'
    ses.meta.source = 'demo_jpan, demo_fasec, demo_magec and migr_imm1ctz tables from Eurostat'

    # EUROSTAT DATASET

    if csv:
        ses.save(os.path.join(DATA_DIR, 'demography_eurostat'))
    if excel:
        ses.save(os.path.join(DATA_DIR, 'demography_eurostat.xlsx'))
    if hdf5:
        ses.save(os.path.join(DATA_DIR, 'demography_eurostat.h5'))

    # EXAMPLE FILES

    years = population.time[2013:2015]
    population = population[years]
    population_narrow = population['Belgium,France'].sum('gender')
    births = births[years]
    deaths = deaths[years]
    immigration = immigration[years]

    # Dataframes (for testing missing axis/values)
    df_missing_axis_name = population.to_frame(fold_last_axis_name=False)
    df_missing_values = population.to_frame(fold_last_axis_name=True)
    df_missing_values.drop([('France', 'Male'), ('Germany', 'Female')], inplace=True)

    if csv:
        examples_dir = os.path.join(DATA_DIR, 'examples')
        population.to_csv(os.path.join(examples_dir, 'population.csv'))
        births.to_csv(os.path.join(examples_dir, 'births.csv'))
        deaths.to_csv(os.path.join(examples_dir, 'deaths.csv'))
        immigration.to_csv(os.path.join(examples_dir, 'immigration.csv'))
        df_missing_axis_name.to_csv(os.path.join(examples_dir, 'population_missing_axis_name.csv'), sep=',', na_rep='')
        df_missing_values.to_csv(os.path.join(examples_dir, 'population_missing_values.csv'), sep=',', na_rep='')
        population_narrow.to_csv(os.path.join(examples_dir, 'population_narrow_format.csv'), wide=False)

    if excel:
        with open_excel(os.path.join(DATA_DIR, 'examples.xlsx'), overwrite_file=True) as wb:
            wb['population'] = population.dump()
            wb['births'] = births.dump()
            wb['deaths'] = deaths.dump()
            wb['immigration'] = immigration.dump()
            wb['population_births_deaths'] = population.dump()
            wb['population_births_deaths']['A9'] = births.dump()
            wb['population_births_deaths']['A17'] = deaths.dump()
            wb['population_missing_axis_name'] = ''
            wb['population_missing_axis_name']['A1'].options().value = df_missing_axis_name
            wb['population_missing_values'] = ''
            wb['population_missing_values']['A1'].options().value = df_missing_values
            # wb['population_narrow_format'] = population_narrow.dump(wide=False)
            wb.save()
        population_narrow.to_excel(os.path.join(DATA_DIR, 'examples.xlsx'), 'population_narrow_format', wide=False)
        Session({'country': population.country, 'gender': population.gender, 'time': population.time,
                 'population': population}).save(os.path.join(DATA_DIR, 'population_only.xlsx'))
        Session({'births': births, 'deaths': deaths}).save(os.path.join(DATA_DIR, 'births_and_deaths.xlsx'))

    if hdf5:
        examples_h5_file = os.path.join(DATA_DIR, 'examples.h5')
        population.to_hdf(examples_h5_file, 'population')
        births.to_hdf(examples_h5_file, 'births')
        deaths.to_hdf(examples_h5_file, 'deaths')
        immigration.to_hdf(examples_h5_file, 'immigration')


if __name__ == '__main__':
    # generate_tests_files()
    generate_example_files()
