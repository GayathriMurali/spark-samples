from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes, LogisticRegressionWithSGD
from numpy import array
from pyspark.mllib.regression import SparseVector
from pyspark.mllib.util import MLUtils

# Load and parse the data
def parsePoint(line):
    dims = vocab_count
    values = [x for x in line.strip().split(' ')]
    features=values[1:]
    f_ind=[]
    f_val=[]
    for i in features:
	#f_ind.append(int(i.split(':')[0])-1)
	f_ind.append(int(i.split(':')[0]))
	f_val.append(float(i.split(':')[1]))
    #print "Index values"+str(len(f_ind))
    #print "Features "+str(len(f_val))
    label = int(values[0])
    if label >=1 and label <= 4:
	label = 0
    elif label >=7 and label <= 10:
	label = 1
    return LabeledPoint(label, SparseVector(dims ,f_ind,f_val))

def g(x):
    print x

dataset_dir = "/Users/gayathrimurali/libraries/imdb-movie-review-dataset/aclImdb"
train_file = "/train/labeledBow.feat"
test_file = "/test/labeledBow.feat"
sc = SparkContext("local", "Classify iMDB Movies")

vocab=sc.textFile("/Users/gayathrimurali/libraries/imdb-movie-review-dataset/aclImdb/imdb.vocab").cache()
vocab_count=vocab.count()
print "Vocab count"+str(vocab_count)

trainData = sc.textFile(dataset_dir + train_file).cache()
print "trainData count: "+str(trainData.count())
parsedData = trainData.map(parsePoint)
model=LogisticRegressionWithSGD.train(parsedData)

testData = sc.textFile(dataset_dir + test_file).cache()
print "testData count: "+str(testData.count())
testparsedData = testData.map(parsePoint)
valuesAndPreds = testparsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))	




#### SCRAP CODE
#examples = MLUtils.loadLibSVMFile(sc, "/Users/gayathrimurali/libraries/glass-dataset/glass.scale")
#print "Examples count: " + str(examples.count())
#print "First example: " + str(examples.first())

#parsedData.foreach(g)

#model=NaiveBayes.train(sc.parallelize(array(parsedData)))
