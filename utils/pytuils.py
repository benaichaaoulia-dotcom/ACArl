class AverageMeter:
    def __init__(self) -> None:
        self.values = []

    def add(self, value_list, num=1):
        # value_list is now a list of per-sample scores
        if isinstance(value_list, list):
            self.values.extend(value_list)
        else:
            self.values.append(value_list)

    def get(self) -> float:
        if len(self.values) == 0:
            return 0
        import numpy as np
        return np.mean(self.values)
    
    def get_std(self) -> float:
        if len(self.values) <= 1:
            return 0.0
        import numpy as np
        return np.std(self.values)

    def __str__(self) -> str:
        return str(self.get())
    
    def __repr__(self) -> str:
        return str(self.get())
