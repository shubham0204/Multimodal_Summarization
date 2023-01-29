#from rouge_metric import PyRouge
import rouge

def evaluate(target_summaries, predicted_summaries):
    evaluator = rouge.Rouge()
    return evaluator.get_scores( predicted_summaries , target_summaries , avg=True )
