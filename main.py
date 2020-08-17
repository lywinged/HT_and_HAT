from skmultiflow.data import FileStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTree
from skmultiflow.trees.hoeffding_adaptive_tree import HAT
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.trees.info_gain_split_criterion import InfoGainSplitCriterion
from skmultiflow.trees.gini_split_criterion import GiniSplitCriterion


# 1. Create a stream
stream = FileStream("elecn.csv")
stream.prepare_for_use()
# 2. Instantiate the HoeffdingTree classifier


count = 100
samples = 110
drift_detector = ADWIN()

nb = NaiveBayes()
ht = HoeffdingTree(split_criterion='info_gain')
hat= HAT()

print("--Evaluating Hoeffding Tree with ADWIN start--")        
while count < samples:
    X, y = stream.next_sample()
    drift_detector.add_element(ht.predict(X)[0] == y[0])
    if drift_detector.detected_change():
      print('Change detected at {}, resetting model'.format(count))
      #print('predict value is {}'.format(ht.predict(X)))
      ht.reset()
    ht.partial_fit(X, y, classes=stream.target_values)
    count += 1


# 3. Setup the evaluator
evaluator = EvaluatePrequential(
                                show_plot = True,
                                max_samples =3000,
                                pretrain_size=500,
                                batch_size =1,
                                n_wait = 200,
                                output_file='results.csv')


# 4. Run evaluation
print("--Evaluating Hoeffding Adative Tree start--")
evaluator.evaluate(stream=stream, model=[hat,ht], model_names=['hat','ht'])
#,'ht','hat'])
print("--Evaluating Hoeffding Adative Tree end--")
print()