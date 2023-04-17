import evaluate

class PRF1Metrics:

    def __init__(self):
        self.f1_score = evaluate.load('f1')
        self.precision_score = evaluate.load('precision')
        self.accuracy_score = evaluate.load('accuracy')
        self.recall_score = evaluate.load("recall")

    def add_batch(self, predictions, references):
        self.f1_score.add_batch(predictions=predictions, references = references)
        self.precision_score.add_batch(predictions=predictions, references=references)
        self.accuracy_score.add_batch(predictions=predictions, references=references)
        self.recall_score.add_batch(predictions=predictions, references=references)

    def compute(self, average="binary"):
        parameters = {"average":average, "zero_division":0}
        f1 = self.f1_score.compute(average=average)
        precision = self.precision_score.compute(**parameters)
        accuracy = self.accuracy_score.compute()
        recall = self.recall_score.compute(**parameters)
        return {**f1, **precision, **accuracy, **recall}

if __name__ == "__main__":
    predictions = [0,0,0,1]
    references = [0,0, 1, 1]
    metrics = PRF1Metrics()
    metrics.add_batch(predictions=predictions, references=references)

    print(metrics.compute())
    metrics.add_batch(predictions=predictions, references=references)
    print(metrics.compute("macro"))
    metrics.add_batch(predictions=predictions, references=references)
    print(metrics.compute("micro"))
