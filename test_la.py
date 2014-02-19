from __future__ import division
from larray import LArray, Axis
import numpy as np

print "numpy", np.__version__

vla = 'A11,A12,A13,A23,A24,A31,A32,A33,A34,A35,A36,A37,A38,A41,A42,A43,A44,' \
      'A45,A46,A71,A72,A73' #.split(',')
wal = 'A25,A51,A52,A53,A54,A55,A56,A57,A61,A62,A63,A64,A65,A81,A82,A83,A84,' \
      'A85,A91,A92,A93' #.split(',')
#bru = ['A21']
bru = 'A21'

#lipro = Axis('lipro', ['P%02d' % i for i in range(1, 16)])
lipro = Axis('lipro', ','.join('P%02d' % i for i in range(1, 16)))
#age = Axis('age', range(116))
age = Axis('age', valuerange(116))
sex = Axis('sex', ['H', 'F'])
# geo = Axis('geo', vla + wal + bru)
geo = Axis('geo', union(vla, wal, bru)) # since these are ordered sets,
# it makes sense to "diasble" non unique keys

print "sex axis", repr(sex)

a = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15).astype(float)
bel = LArray(a, axes=(age, geo, sex, lipro))

# ages1_5_9 = age.group(1, 5, 9)
ages1_5_9 = age.group('1,5,9')
print ages1_5_9

bel020 = bel.filter(age=ages1_5_9, lipro='P01,P02') # no axes order => use parent!
print bel020.info()
#print bel020
# bel020 = bel.filter(age=[1, 5, 9], lipro='P01,P02') # no axes order => use parent!
bel020 = bel.filter(age='1,5,9', lipro='P01,P02') # no axes order => use
# parent!
bel020 = bel.filter(sex='H', lipro='P01,P02') # 3 dimensions (sex is gone)
print bel020.info()

# aggbel = bel.sum(0, 2) # 2d (geo, lipro)
# print "aggbel.shape", aggbel.shape
aggbel = bel.sum(age, sex) # 2d (geo, lipro)
print "aggbel.shape", aggbel.shape

vla = geo.group(vla, name='vla')
wal = geo.group(wal, name='wal')
bru = geo.group('A21', name='bru')
belgium = geo.group(':', name='belgium')
# belgium = geo[:]
# belgium = geo.all() #'belgium')


#belgium = vla + wal + bru # equivalent
#wal_bru = belgium - vla
#wal_bru = wal + bru # equivalent

aggbelvla = aggbel[vla]
print "aggbel[vla]", aggbelvla.info()
print aggbelvla

# 2d (geo, lipro) -> 2d (geo', lipro)
reg = aggbel.sum(geo=(vla, wal, bru, belgium))
print "reg", reg.info()
print reg

# regsum = reg.sum(lipro=('P01', 'P02', lipro.all()))
regsum = reg.sum(lipro='P01,P02,P03,:')
print regsum

regvla = reg[vla]
# print reg['vla,P15']
print "regvla", regvla.info()
print regvla
print regvla.info()

regvlap03 = regvla['P03']
print "regvlap03 axes", regvlap03.info()
print regvlap03

# child = age[:17] # stop bound is inclusive !
child = age.group(':17') # stop bound is inclusive !
# working = age[18:64] # create a ValueGroup(Axis(age), [18, ..., 64], '18:64')
working = age.group('18:64') # create a ValueGroup(Axis(age), [18, ..., 64],
# '18:64')
# retired = age[65:]
retired = age.group('65:')
#arr3x = geo.group('A3*') # * match one or more chars
#arr3x = geo.group('A3?') # ? matches one char (equivalent in this case)
#arr3x = geo.seq('A31', 'A38')
# arr3x = geo['A31':'A38'] # not equivalent! (if A22 is between A31 and A38)

test = bel.filter(age=child)
print "test", test.info()
# test = bel.filter(age=age[:17]).filter(geo=belgium)
test = bel.filter(age=':17').filter(geo=belgium)
print test.info()
# test = bel.filter(age=range(18))
# print test.info()

# ages = bel.sum(age=(child, 5, working, retired))
ages = bel.sum(age=(child, '5:10', working, retired))
print ages.info()
ages2 = ages.filter(age=child)
print ages2.info()

#print "ages.filter", ages.filter(age=:17) # invalid syntax
#print "ages.filter(age=':17')", ages.filter(age=':17')
#print "ages.filter(age=slice(17))", ages.filter(age=slice(17))

total = reg.sum()            # total (including duplicates like belgium?)
print "total", total
# total (including duplicates like belgium?)
total = reg.sum(geo, lipro)
print "total", total

ratio = reg.ratio(geo, lipro)
print "reg.ratio(geo, lipro)"
print ratio.info()
print ratio
print ratio.sum()

x = bel.sum(age) # 3d matrix
print "sum(age)"
print x.info()

x = bel.sum(lipro, geo) # 2d matrix
print "sum(lipro, geo)"
print x.info()

x = bel.sum(lipro, geo=geo.all()) # the same 2d matrix?
x = bel.sum(lipro, geo=':') # the same 2d matrix?
x = bel.sum(lipro, geo='A11:A33') # include everything between the two labels
#  (it can include 'A63')
print "sum(lipro, geo=geo.all())"
print "sum(lipro, geo=geo.all())"
print x.info()

x = bel.sum(lipro, sex='H') # a 3d matrix (sex dimension of size 1)
print "sum(lipro, sex='H')"
print x.info()

# x = bel.sum(lipro, sex=(['H'], ['H', 'F'])) # idem
x = bel.sum(lipro, sex=('H', 'H,F')) # idem
x = bel.sum(lipro, sex='H;H,F') # <-- abbreviation
print "sum(lipro, sex=('H',))"
print x.info()

x = bel.sum(lipro, geo=(geo.all(),)) # 3d matrix (geo dimension of size 1)
print "sum(lipro, geo=(geo.all(),))"
print x.info()
#print bel.sum(lipro, geo=(vla, wal, bru)) # 3d matrix

#bel.sum(lipro, geo=(vla, wal, bru), sex) # <-- not allowed!!! (I think we can live with that)

newbel = bel.reorder(age, geo, sex, lipro)

#print newbel[:10,"A11",:,"P02"]

# arithmetic tests
small_data = np.random.randn(2, 15)
small = LArray(small_data, axes=(sex, lipro))
print small
print small + small
print small * 2
print 2 * small
print small + 1
print 1 + small
print 30 / small
print 30 / (small + 1)
print small / small
small_int = LArray(small_data, axes=(sex, lipro))
print "truediv"
print small_int / 2
print "floordiv"
print small_int // 2

# excel export
print "excel export",
reg.to_excel('c:\\tmp\\reg.xlsx', '_')
#ages.to_excel('c:/tmp/ages.xlsx')
print "done"

# test plotting
#small_h = small['H']
#small_h.plot(kind='bar')
#small_h.plot()
#small_h.hist()

#large_data = np.random.randn(1000)
#tick_v = np.random.randint(ord('a'), ord('z'), size=1000)
#ticks = [chr(c) for c in tick_v]
#large_axis = Axis('large', ticks)
#large = LArray(large_data, axes=[large_axis])
#large.plot()
#large.hist()