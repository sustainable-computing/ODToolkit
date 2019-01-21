import numpy as np
import itertools

class GridSearch():
    def __init__(self, 
                 dataset,
                 model,
                 metric,
                 possible_params):
                 
        self.dataset = dataset
        self. model = model
        self.metric = metric
        self.possible_params
    
    def tuned_params(self):
        return self.get_best_parameters(parameter_list=self.get_params_permut(self.possible_params),
                                        dataset_list=self.get_k_fold_partitions(self.dataset),
                                        estimator=self.model,
                                        metric=self.metric)

    def get_params_permut(self,
                          possible_params):
                          
        keys, values = zip(*possible_params.items())
        params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        return params_list

    def get_k_fold_partitions(self,
                  dataset,
                  K=5):
        dataset_list = []
        header = dataset.header
        for k in range(K):
            train_features = [x for i, x in enumerate(dataset.data) if i % K != k]
            train_occupancy = [x for i, x in enumerate(dataset.occupancy) if i % K != k]
            
            test_features = [x for i, x in enumerate(dataset.data) if i % K == k]
            test_occupancy = [x for i, x in enumerate(dataset.occupancy) if i % K == k]
            
            daset_list.append( (Dataset().add_room(room_data=train_features, room_occupancy=train_occupancy, header=header),
                                Dataset().add_room(room_data=test_features, room_occupancy=test_occupancy, header=header)) )
        return dataset_list
        
    def get_best_parameters(self,
                            metric,
                            parameter_list,
                            dataset_list,
                            estimator):
                            
        best_parameters = {}
        best_value = 0.0
        for parameter in parameter_list:
            values = []
            for (train, test) in dataset_list:
                result = estimator(train, test, parameter).run()
                values.add(metric(test.occupancy, result))
            avg = np.mean(values)
            if avg > best_value:
                best_parameters = parameter
                best_value = avg
        return best_parameters
    
    
x = [(1,2), (3,4)]
for (i,j) in x:
    print(str(i)+' '+str(j))
