from sklearn.datasets import load_iris
import numpy as np
import math

def _mean(slength, swidth, plength, pwidth):
    a = np.mean(slength)
    b = np.mean(swidth)
    c = np.mean(plength)
    d = np.mean(pwidth)
    return a, b, c, d

def _var(slength, swidth, plength, pwidth):
    a = np.std(slength)
    b = np.std(swidth)
    c = np.std(plength)
    d = np.std(pwidth)
    return a, b, c, d

def decFormat(x):
    return ('%.4f' % x).rstrip('0').rstrip('.')

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
target_names = data['target_names']
labels = target_names[target]
is_setosa = (labels == 'setosa')
is_versicolor = (labels == 'versicolor')
is_virginica = (labels == 'virginica')

sepal_length = features[:, 0]
sepal_width = features[:,1]
petal_length = features[:, 2]
petal_width = features[:,3]

# setosa
slength_setosa = sepal_length[is_setosa]
swidth_setosa = sepal_width[is_setosa]
plength_setosa = petal_length[is_setosa]
pwidth_setosa = petal_width[is_setosa]

# versicolor
slength_versicolor = sepal_length[is_versicolor]
swidth_versicolor = sepal_width[is_versicolor]
plength_versicolor = petal_length[is_versicolor]
pwidth_versicolor = petal_width[is_versicolor]

# virginica
slength_virginica = sepal_length[is_virginica]
swidth_virginica = sepal_width[is_virginica]
plength_virginica = petal_length[is_virginica]
pwidth_virginica = petal_width[is_virginica]

# mean
mean_slength_setosa, mean_swidth_setosa, mean_plength_setosa, mean_pwidth_setosa = \
    _mean(slength_setosa, swidth_setosa, plength_setosa, pwidth_setosa)
mean_slength_versicolor, mean_swidth_versicolor, mean_plength_versicolor, mean_pwidth_versicolor =\
    _mean(slength_versicolor, swidth_versicolor, plength_versicolor, pwidth_versicolor)
mean_slength_virginica, mean_swidth_virginica, mean_plength_virginica, mean_pwidth_virginica = \
    _mean(slength_virginica, swidth_virginica, plength_virginica, pwidth_virginica)

# var
var_slength_setosa, var_swidth_setosa, var_plength_setosa, var_pwidth_setosa = \
    _var(slength_setosa, swidth_setosa, plength_setosa, pwidth_setosa)
var_slength_versicolor, var_swidth_versicolor, var_plength_versicolor, var_pwidth_versicolor = \
    _var(slength_versicolor, swidth_versicolor, plength_versicolor, pwidth_versicolor)
var_slength_virginica, var_swidth_virginica, var_plength_virginica, var_pwidth_virginica = \
    _var(slength_virginica, swidth_virginica, plength_virginica, pwidth_virginica)

sl = float(input("Masukan Sepal Length : "))
sw = float(input("Masukan Sepal Width : "))
pl = float(input("Masukan Petal Length : "))
pw = float(input("Masukan Petal Width : "))

test = [sl,sw,pl,pw]

setosa_A = calculateProbability(test[0], mean_slength_setosa, var_slength_setosa) \
           * calculateProbability(test[1], mean_swidth_setosa, var_swidth_setosa) \
           * calculateProbability(test[2], mean_plength_setosa, var_plength_setosa) \
           * calculateProbability(test[3], mean_pwidth_setosa, var_pwidth_setosa)
versicolor_A = calculateProbability(test[0], mean_slength_versicolor, var_slength_versicolor)\
               * calculateProbability(test[1], mean_swidth_versicolor, var_swidth_versicolor) \
               * calculateProbability(test[2], mean_plength_versicolor, var_plength_versicolor) \
               * calculateProbability(test[3], mean_pwidth_versicolor, var_pwidth_versicolor)
vir_A = calculateProbability(test[0], mean_slength_virginica, var_slength_virginica) \
        * calculateProbability(test[1], mean_swidth_virginica, var_swidth_virginica) \
        * calculateProbability(test[2], mean_plength_virginica, var_plength_virginica) \
        * calculateProbability(test[3], mean_pwidth_virginica, var_pwidth_virginica)

temp = []
temp.append([setosa_A, 'Setosa'])
temp.append([versicolor_A, 'Versicolor'])
temp.append([vir_A, 'Virginica'])

maks = max(temp)

a = ('Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width')

print('----------------------------------------------')
print('Setosa')
print('------')
print('     '+'\t  '.join(a))
print('Mean',  str(mean_slength_setosa), '\t\t\t ', str(mean_swidth_setosa), '\t\t ', str(mean_plength_setosa),
      '\t\t ',str(mean_pwidth_setosa))
print('Var ',  str(decFormat(var_slength_setosa)), '\t\t ', str(decFormat(var_swidth_setosa)), '\t\t ',
      str(decFormat(var_plength_setosa)),'\t\t ',str(decFormat(var_pwidth_setosa)))

print('----------------------------------------------')
print('Versicolor')
print('----------')
print('     '+'\t  '.join(a))
print('Mean',  str(mean_slength_versicolor), '\t\t\t ', str(decFormat(mean_swidth_versicolor)), '\t\t\t '
      , str(decFormat(mean_plength_versicolor)),'\t\t\t ',str(decFormat(mean_pwidth_versicolor)))
print('Var ',  str(decFormat(var_slength_versicolor)), '\t\t\t ', str(decFormat(var_swidth_versicolor)), '\t\t ',
      str(decFormat(var_plength_versicolor)),'\t\t ',str(decFormat(var_pwidth_versicolor)))

print('----------------------------------------------')
print('Virginica')
print('---------')
print('     '+'\t  '.join(a))
print('Mean',  str(decFormat(mean_slength_virginica)), '\t\t\t ', str(mean_swidth_virginica), '\t\t '
      , str(decFormat(mean_plength_virginica)),'\t\t ',str(decFormat(mean_pwidth_virginica)))
print('Var ',  str(decFormat(var_slength_virginica)), '\t\t ', str(decFormat(var_swidth_virginica)), '\t\t ',
      str(decFormat(var_plength_virginica)),'\t\t ',str(decFormat(var_pwidth_virginica)))


print('----------------------------------------------')
print('Probabilitas')
print('Setosa: ', setosa_A,'|| Versicolor:  ', versicolor_A, '|| Virginica : ', vir_A)
print('----------------------------------------------')

print(maks[1])