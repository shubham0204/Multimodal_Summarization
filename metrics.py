import evaluate

rouge = evaluate.load('rouge')

def evaluate(target_summaries, predicted_summaries):
  return rouge.compute( predictions=predicted_summaries, references=target_summaries)