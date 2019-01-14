class NormalModel:
    def __init__(self,
                 train,
                 test):
        self.train = train
        self.test = test
        self.models = {}
        self.get_all_model()

    def get_all_model(self):
        for model in NormalModel.__subclasses__():
            self.models[model.__name__] = model(self.train, self.test)

    def run_all_model(self):
        for model in self.models.keys():
            result = self.models[model].run()
            # Evaluate result
            print(model, result)


class DomainAdaptiveModel:
    def __init__(self,
                 source,
                 target_retrain,
                 target_test):
        self.source = source
        self.target_retrain = target_retrain
        self.target_test = target_test
        self.models = {}
        self.get_all_model()

    def get_all_model(self):
        for model in DomainAdaptiveModel.__subclasses__():
            self.models[model.__name__] = model(self.source, self.target_retrain, self.target_test)

    def run_all_model(self):
        for model in self.models.keys():
            result = self.models[model].run()
            # Evaluate result
            print(model, result)
