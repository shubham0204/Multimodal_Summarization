#from rouge_metric import PyRouge
import rouge

def evaluate(target_summaries, predicted_summaries):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                               max_n=4,
                               limit_length=False,
                               length_limit=1000,
                               length_limit_type='words',
                               apply_avg=True,
                               apply_best=False,
                               alpha=0.2,
                               weight_factor=1.2,
                               stemming=True)
    return evaluator.get_scores( predicted_summaries , target_summaries )

