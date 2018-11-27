import numpy as np
import multiprocessing as mp
import time


class ParallelContainer:

    def __init__(self, **kwargs):

        prop_defaults = {
            'num_processors': None,
            'num_splits': 1000,
            'verbose': -1,
            'timer': False
        }

        for (prop, default) in prop_defaults.items():
                setattr(self, prop, kwargs.get(prop, default))

    def _prep_parallel(self, length, binary_args, sensitivity_args, verbose):
        if self.num_processors == -1:
            self.num_processors = mp.cpu_count()

        split_val = int(np.ceil(length/self.num_splits))
        split_inds = [self.num_splits*i for i in np.arange(1, split_val)]

        inds_split_all = np.split(np.arange(length), split_inds)

        self.args = []
        for i, ind_split in enumerate(inds_split_all):
            trans_args = []
            for arg in binary_args:
                try:
                    trans_args.append(arg[ind_split])
                except TypeError:
                    trans_args.append(arg)

            self.args.append((i,) + tuple(trans_args) + sensitivity_args + (verbose,))

        return

    def _run_parallel(self, para_func):
        if self.timer:
            start_timer = time.time()

        # for testing
        # check = parallel_snr_func(*self.args[10])
        # import pdb
        # pdb.set_trace()

        print('numprocs', self.num_processors)
        with mp.Pool(self.num_processors) as pool:
            print('start pool with {} processors: {} total processes.\n'.format(
                    self.num_processors, len(self.args)))

            results = [pool.apply_async(para_func, arg) for arg in self.args]
            out = [r.get() for r in results]
            out = {key: np.concatenate([out_i[key] for out_i in out]) for key in out[0].keys()}
        if self.timer:
            print("SNR calculation time:", time.time()-start_timer)
        return out
