from collections import defaultdict


class BaseTracker:
    def __init__(self, refiner):
        # attach the tracker to the refiner
        refiner.tracker = self

        # attach the tracker to the optimizer(s)
        opts = refiner.optimizer
        opts = opts if isinstance(opts, (tuple, list)) else [opts]
        for opt in opts:
            opt.logging_fn = self.log_optim_iter

    def log_dense(self, **args):
        raise NotImplementedError

    def log_optim_done(self, **args):
        raise NotImplementedError

    def log_optim_iter(self, **args):
        raise NotImplementedError


class SimpleTracker(BaseTracker):
    def __init__(self, refiner):
        super().__init__(refiner)

        self.dense = defaultdict(dict)
        self.costs = []
        self.T = []
        self.dt = []
        self.p3d = None
        self.p3d_ids = None
        self.num_iters = []

    def log_dense(self, **args):
        feats = [f.cpu() for f in args['features']]
        weights = [w.cpu()[0] for w in args['weight']]
        data = (args['image'], feats, weights)
        self.dense[args['name']][args['image_scale']] = data

    def log_optim_done(self, **args):
        self.p3d = args['p3d']
        self.p3d_ids = args['p3d_ids']

    def log_optim_iter(self, **args):
        if args['i'] == 0:  # new scale or level
            self.costs.append([])
            self.T.append(args['T_init'].cpu())
            self.num_iters.append(None)

        valid = args['valid'].float()
        cost = ((valid*args['cost']).sum(-1)/valid.sum(-1))

        self.costs[-1].append(cost.cpu().numpy())
        self.dt.append(args['T_delta'].magnitude()[1].cpu().numpy())
        self.num_iters[-1] = args['i']+1
        self.T.append(args['T'].cpu())
