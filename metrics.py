from rouge_metric import PyRouge
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

def compute_rouge_1( target_summaries , predicted_summaries ):
    rouge = PyRouge( rouge_n=1 )
    # Hypothesis -> predicted_summaries
    # References -> target_summaries
    scores = rouge.evaluate( predicted_summaries , target_summaries )
    # Return only the f1 score
    return scores[ 'rouge-1' ]

def compute_rouge_2( target_summaries , predicted_summaries ):
    rouge = PyRouge( rouge_n=2 )
    # Hypothesis -> predicted_summaries
    # References -> target_summaries
    scores = rouge.evaluate( predicted_summaries , target_summaries )
    # Return only the f1 score
    return scores[ 'rouge-2' ][ 'f' ]

def compute_rouge_LCS( target_summaries , predicted_summaries ):
    rouge = PyRouge( rouge_l=True )
    # Hypothesis -> predicted_summaries
    # References -> target_summaries
    scores = rouge.evaluate( predicted_summaries , target_summaries )
    # Return only the f1 score
    return scores[ 'rouge-l' ][ 'f' ]

