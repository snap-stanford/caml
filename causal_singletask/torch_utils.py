""" Util functions and classes for train.py script"""

class BestValueLogger:
    """
    Logs the best value and step for a list of metrics (names)
    """
    def __init__(self, metrics: dict={}):
        """
        metrics: dict with metric names as keys and values representing
        bools indicating whether larger values are better (True)
        or worse (False) for each metric.
        """
        self.names = list(metrics.keys())
        self.increasing = metrics 
        # dict that maps metric name to bool if the 
        # metric needs to be increased or not 
        self.best_values = {'_'.join(['best', name, suffix]): 
                None for name in self.names for suffix in ['step', 'value']}
    
    def __call__(self, name: str, value: float, step: int, increasing: bool): 
        """
        We assume that values refer to the names at init 
        """
        if not self._is_metric_registered(name):
            self._add_metric(name, increasing)
        
        return self._update_metric(name, value, step, increasing)
        
    def _is_metric_registered(self, name):
        """check if current metric is already in dict """
        key = '_'.join(['best', name, 'value'])
        return key in self.best_values.keys()

    def _add_metric(self, name, increasing):
        """ Initialize empty metric, adding it to best_value dict
        """
        self.best_values.update( 
            {'_'.join(['best', name, suffix]): 
                None for suffix in ['step', 'value']}
        )
        self.increasing.update(
            {name: increasing}
        )
    def _update_metric(self, name, value, step, increasing):
        """
        Check if current value is better than best before, if yes, update.
        """
        # sanity check that step is larger than previous best:
        newBestValue = False

        self._sanity_check_non_decreasing_steps(name,step)
        assert increasing == self.increasing[name] # check that we use increasing 
        # consistently across calls
        larger_better = int(increasing)
        larger_better = 2*larger_better - 1 # [1,-1]
        old_value = self.best_values['best_' + name + '_value']
        # init if empty at start:
        if old_value is None:
            old_value = 1e15 * (-larger_better)
        if larger_better * (value - old_value) > 0:
            self.best_values['best_' + name+'_value'] = value
            self.best_values['best_' + name+'_step'] = step

            newBestValue = True

        return newBestValue

    def _sanity_check_non_decreasing_steps(self, name, step):
        if self.best_values['best_' + name+'_step']:
            assert self.best_values['best_' + name+'_step'] < step
            
class EarlyStopping(BestValueLogger):
    def __init__(self, name, increasing=False, patience=5):
        super(EarlyStopping, self).__init__(
            [name],
            [increasing]
        )
        self.patience = patience
        self.count = 0

    def __call__(self, values: list, step: int): 
        """
        We assume that values refer to the names at init 
        """
        assert len(values) == len(self.names)
        data = dict(zip(self.names, values))

        for name, value in data.items():
            stopped = self._update_metric(name, value, step)
            if stopped:
                return True
        return False

    def _update_metric(self, name, value, step):
        """
        Check if current value is better than best before, if yes, reset count,
        otherwise increasee.
        """
        larger_better = int(self.increasing[name])
        larger_better = 2*larger_better - 1 # [1,-1]
        old_value = self.best_values['best_' + name + '_value']
        # init if empty at start:
        if old_value is None:
            old_value = 1e9 * (-larger_better)
        # if we improve, reset counter:
        if larger_better * (value - old_value) > 0:
            self.best_values['best_' + name+'_value'] = value
            self.best_values['best_' + name+'_step'] = step
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                return True
        return False
 



